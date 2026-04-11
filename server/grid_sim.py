"""
BlackoutShield Grid Simulator (Merged)
IEEE 5-Bus topology with DC power flow, thermal cascades, SCADA spoofing.
Combines fog-of-war cache + lstsq islanding fix + full environment API.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Bus:
    id: int
    name: str
    voltage: float = 1.0
    generation: float = 0.0
    load: float = 0.0
    is_slack: bool = False
    is_hospital: bool = False


@dataclass
class Line:
    id: int
    from_bus: int
    to_bus: int
    impedance: float
    thermal_limit: float
    status: bool = True
    flow: float = 0.0


@dataclass
class SCADASpoof:
    bus_id: int
    fake_voltage: float
    active: bool = True


class GridSimulator:
    """
    5-Bus IEEE power grid simulator.

    Topology:
        GEN0 (slack, 1.0V) --0.1Ω-- BUS1 --0.15Ω-- BUS4 (HOSPITAL)
                                      |
                                    0.2Ω
                                      |
                                    BUS2 --0.25Ω-- BUS3 (vulnerable)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.time_step = 0
        self._build_topology()
        self.spoofs: List[SCADASpoof] = []
        self.tripped_lines: List[int] = []
        self.cascade_count = 0
        # Fog of War: cached SCADA readings (may be stale/spoofed)
        self.last_scada_voltages: Dict[int, float] = {i: 1.0 for i in range(5)}

    def _build_topology(self):
        """Initialize the 5-bus system."""
        self.buses = {
            0: Bus(0, "GEN0", voltage=1.0, generation=2.5, is_slack=True),
            1: Bus(1, "BUS1", voltage=1.0, load=0.5),
            2: Bus(2, "BUS2", voltage=1.0, load=0.4),
            3: Bus(3, "BUS3", voltage=1.0, load=0.6),
            4: Bus(4, "BUS4", voltage=1.0, load=0.8, is_hospital=True),
        }
        self.lines = {
            0: Line(0, 0, 1, impedance=0.1, thermal_limit=3.0),
            1: Line(1, 1, 4, impedance=0.15, thermal_limit=1.5),
            2: Line(2, 1, 2, impedance=0.2, thermal_limit=1.5),
            3: Line(3, 2, 3, impedance=0.25, thermal_limit=1.0),
        }
        self.num_buses = len(self.buses)
        self.num_lines = len(self.lines)

    def reset(self):
        """Reset grid to initial state."""
        self.rng = np.random.RandomState(self.seed)
        self.time_step = 0
        self._build_topology()
        self.spoofs = []
        self.tripped_lines = []
        self.cascade_count = 0
        self.last_scada_voltages = {i: 1.0 for i in range(5)}
        self.solve_power_flow()

    def _build_b_matrix(self) -> np.ndarray:
        """Build susceptance (B) matrix from active lines."""
        B = np.zeros((self.num_buses, self.num_buses))
        for line in self.lines.values():
            if not line.status:
                continue
            b_val = 1.0 / line.impedance
            i, j = line.from_bus, line.to_bus
            B[i, i] += b_val
            B[j, j] += b_val
            B[i, j] -= b_val
            B[j, i] -= b_val
        return B

    def _is_bus_connected(self, bus_id: int) -> bool:
        """Check if a bus has any active line connection."""
        return any(
            (l.from_bus == bus_id or l.to_bus == bus_id) and l.status
            for l in self.lines.values()
        )

    def solve_power_flow(self):
        """
        DC Power Flow: P = B @ theta
        Uses lstsq fallback for islanded/singular grids.
        Disconnected buses drop to 0V.
        """
        B = self._build_b_matrix()
        P = np.array([b.generation - b.load for b in self.buses.values()])

        non_slack = list(range(1, self.num_buses))
        B_red = B[np.ix_(non_slack, non_slack)]
        P_red = P[non_slack]

        # Islanding-safe solver
        try:
            if np.linalg.cond(B_red) < 1 / np.finfo(B_red.dtype).eps:
                theta_red = np.linalg.solve(B_red, P_red)
            else:
                theta_red = np.linalg.lstsq(B_red, P_red, rcond=None)[0]
        except np.linalg.LinAlgError:
            theta_red = np.zeros(len(non_slack))

        theta = np.zeros(self.num_buses)
        theta[1:] = theta_red

        # Compute line flows
        for line in self.lines.values():
            if line.status:
                line.flow = (theta[line.from_bus] - theta[line.to_bus]) / line.impedance
            else:
                line.flow = 0.0

        # Update bus voltages
        for bus in self.buses.values():
            if bus.is_slack:
                bus.voltage = 1.0
            elif not self._is_bus_connected(bus.id):
                # Islanded bus: if it has local generation, maintain voltage
                if bus.generation > 0 and bus.load > 0:
                    bus.voltage = min(1.05, bus.generation / bus.load)
                elif bus.generation > 0:
                    bus.voltage = 1.0
                else:
                    bus.voltage = 0.0  # No gen = blackout
            else:
                drop = abs(theta[bus.id]) * 0.07 + self.rng.normal(0, 0.002)
                bus.voltage = max(0.0, min(1.1, 1.0 - drop))

    def get_line_flows(self) -> Dict[int, float]:
        return {lid: line.flow for lid, line in self.lines.items()}

    def get_thermal_status(self) -> Dict[int, float]:
        """Returns thermal ratio (flow/limit) per line. >1.0 = overloaded."""
        return {
            lid: abs(line.flow) / line.thermal_limit if line.status and line.thermal_limit > 0 else 0.0
            for lid, line in self.lines.items()
        }

    def get_bus_voltages(self) -> Dict[int, float]:
        """True physical voltages (not visible to agent without scanning)."""
        return {b.id: b.voltage for b in self.buses.values()}

    def get_scada_voltages(self) -> Dict[int, float]:
        """The 'lying' sensor data — may be spoofed. Updates fog-of-war cache."""
        voltages = self.get_bus_voltages()
        for spoof in self.spoofs:
            if spoof.active:
                voltages[spoof.bus_id] = spoof.fake_voltage
        self.last_scada_voltages = voltages.copy()
        return voltages

    def _check_cascade(self):
        """Recursive thermal cascade: trip overloaded lines, re-solve, repeat."""
        tripped_this_round = False
        for lid, line in self.lines.items():
            if line.status and abs(line.flow) > line.thermal_limit:
                line.status = False
                line.flow = 0.0
                self.tripped_lines.append(lid)
                tripped_this_round = True
                self.cascade_count += 1

        if tripped_this_round:
            self.solve_power_flow()
            self._check_cascade()

    def trip_line(self, line_id: int) -> bool:
        """Trip (disconnect) a line."""
        if line_id in self.lines and self.lines[line_id].status:
            self.lines[line_id].status = False
            self.lines[line_id].flow = 0.0
            self.tripped_lines.append(line_id)
            self.solve_power_flow()
            self._check_cascade()
            return True
        return False

    def restore_line(self, line_id: int) -> bool:
        """Restore a tripped line."""
        if line_id in self.lines and not self.lines[line_id].status:
            self.lines[line_id].status = True
            if line_id in self.tripped_lines:
                self.tripped_lines.remove(line_id)
            self.solve_power_flow()
            return True
        return False

    def inject_spoof(self, bus_id: int, fake_voltage: float):
        """Inject a SCADA spoofing attack on a bus."""
        self.spoofs = [s for s in self.spoofs if s.bus_id != bus_id]
        self.spoofs.append(SCADASpoof(bus_id=bus_id, fake_voltage=fake_voltage))

    def remove_spoof(self, bus_id: int):
        """Remove spoof from a bus (after successful scan)."""
        self.spoofs = [s for s in self.spoofs if s.bus_id != bus_id]

    def scan_bus(self, bus_id: int) -> Dict:
        """Scan a bus to reveal true readings (bypasses SCADA spoofing)."""
        bus = self.buses[bus_id]
        is_spoofed = any(s.bus_id == bus_id and s.active for s in self.spoofs)
        scada_v = self.get_scada_voltages()[bus_id]
        return {
            "bus_id": bus_id,
            "true_voltage": bus.voltage,
            "scada_voltage": scada_v,
            "is_spoofed": is_spoofed,
            "load": bus.load,
            "generation": bus.generation,
            "connected": self._is_bus_connected(bus_id),
        }

    def isolate_bus(self, bus_id: int):
        """Isolate a bus by tripping all connected lines."""
        for line in self.lines.values():
            if (line.from_bus == bus_id or line.to_bus == bus_id) and line.status:
                self.trip_line(line.id)

    def shed_load(self, bus_id: int, amount: float = 0.2):
        """Reduce load on a bus (load shedding)."""
        if bus_id in self.buses:
            self.buses[bus_id].load = max(0.0, self.buses[bus_id].load - amount)
            self.solve_power_flow()

    def reinforce_bus(self, bus_id: int, amount: float = 0.1):
        """Increase generation near a bus."""
        if bus_id in self.buses:
            self.buses[bus_id].generation += amount
            self.solve_power_flow()

    def get_hospital_power(self) -> float:
        """Get hospital bus (BUS4) voltage/power level."""
        return self.buses[4].voltage

    def get_load_served(self) -> float:
        """Calculate fraction of total load being served."""
        total_load = sum(b.load for b in self.buses.values())
        if total_load == 0:
            return 1.0
        served = sum(b.load for b in self.buses.values() if b.voltage > 0.8)
        return served / total_load

    def get_state(self) -> Dict:
        """Full internal state snapshot."""
        return {
            "time_step": self.time_step,
            "buses": {
                bid: {
                    "name": b.name,
                    "voltage": round(b.voltage, 4),
                    "generation": b.generation,
                    "load": b.load,
                    "is_hospital": b.is_hospital,
                    "connected": self._is_bus_connected(bid),
                }
                for bid, b in self.buses.items()
            },
            "lines": {
                lid: {
                    "from": l.from_bus,
                    "to": l.to_bus,
                    "flow": round(l.flow, 4),
                    "thermal_ratio": round(abs(l.flow) / l.thermal_limit, 4) if l.status and l.thermal_limit > 0 else 0.0,
                    "status": l.status,
                }
                for lid, l in self.lines.items()
            },
            "spoofs_active": len([s for s in self.spoofs if s.active]),
            "tripped_lines": self.tripped_lines.copy(),
            "cascade_count": self.cascade_count,
            "hospital_power": round(self.get_hospital_power(), 4),
            "load_served": round(self.get_load_served(), 4),
            "fog_of_war": self.last_scada_voltages.copy(),
        }

    def step(self):
        """Advance one time step (called by environment)."""
        self.time_step += 1
        for bus in self.buses.values():
            if not bus.is_slack and bus.load > 0:
                bus.load += self.rng.normal(0, 0.01)
                bus.load = max(0.1, bus.load)
        self.solve_power_flow()
