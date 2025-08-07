import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

@dataclass
class Process:
    """Represents a process in the system"""
    pid: int
    arrival_time: int
    burst_time: int
    priority: int
    remaining_time: int = 0
    waiting_time: int = 0
    turnaround_time: int = 0
    completion_time: int = 0
    start_time: int = -1
    response_time: int = -1
    
    def __post_init__(self):
        self.remaining_time = self.burst_time

@dataclass 
class Resource:
    """Represents a system resource"""
    rid: int
    name: str
    total_instances: int
    available_instances: int
    allocated: Dict[int, int] = None  # pid -> allocated count
    max_need: Dict[int, int] = None   # pid -> maximum need
    
    def __post_init__(self):
        if self.allocated is None:
            self.allocated = {}
        if self.max_need is None:
            self.max_need = {}

@dataclass
class ExecutionStep:
    """Represents a single step in algorithm execution"""
    time: int
    action: str
    process_id: int
    details: str
    queue_state: List[int]
    process_states: Dict[int, str]

class CPUScheduler:
    """Implements all major CPU scheduling algorithms with step-by-step tracking"""
    
    def __init__(self):
        self.processes = []
        self.gantt_chart = []
        self.current_time = 0
        self.algorithm_results = {}
        self.execution_steps = []
    
    def add_process(self, process: Process):
        """Add a process to the scheduler"""
        self.processes.append(process)
    
    def reset(self):
        """Reset scheduler state"""
        self.gantt_chart = []
        self.current_time = 0
        self.execution_steps = []
        for process in self.processes:
            process.remaining_time = process.burst_time
            process.waiting_time = 0
            process.turnaround_time = 0
            process.completion_time = 0
            process.start_time = -1
            process.response_time = -1
    
    def add_step(self, action: str, process_id: int, details: str, queue_state: List[int] = None, process_states: Dict[int, str] = None):
        """Add an execution step for tracking"""
        if queue_state is None:
            queue_state = []
        if process_states is None:
            process_states = {}
        
        step = ExecutionStep(
            time=self.current_time,
            action=action,
            process_id=process_id,
            details=details,
            queue_state=queue_state.copy(),
            process_states=process_states.copy()
        )
        self.execution_steps.append(step)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not self.processes:
            return {}
        
        total_waiting = sum(p.waiting_time for p in self.processes)
        total_turnaround = sum(p.turnaround_time for p in self.processes)
        total_response = sum(p.response_time for p in self.processes if p.response_time != -1)
        
        n = len(self.processes)
        max_completion = max(p.completion_time for p in self.processes) if self.processes else 0
        total_burst = sum(p.burst_time for p in self.processes)
        
        return {
            'avg_waiting_time': total_waiting / n,
            'avg_turnaround_time': total_turnaround / n,
            'avg_response_time': total_response / n if total_response > 0 else 0,
            'cpu_utilization': (total_burst / max_completion * 100) if max_completion > 0 else 0,
            'throughput': n / max_completion if max_completion > 0 else 0
        }
    
    def fcfs(self) -> Tuple[List, Dict]:
        """First-Come First-Served scheduling with step tracking"""
        self.reset()
        self.add_step("INIT", -1, "Starting FCFS Algorithm")
        
        # Sort by arrival time
        sorted_processes = sorted(self.processes, key=lambda p: p.arrival_time)
        self.add_step("SORT", -1, f"Processes sorted by arrival time: {[p.pid for p in sorted_processes]}")
        
        for i, process in enumerate(sorted_processes):
            # Wait for process arrival
            if self.current_time < process.arrival_time:
                self.add_step("WAIT", -1, f"CPU idle from time {self.current_time} to {process.arrival_time}", 
                             [], {p.pid: "Waiting" if p.arrival_time <= process.arrival_time else "Not arrived" for p in self.processes})
                self.current_time = process.arrival_time
            
            # Set start and response time
            process.start_time = self.current_time
            process.response_time = self.current_time - process.arrival_time
            
            self.add_step("START", process.pid, 
                         f"P{process.pid} starts execution at time {self.current_time} (Response time: {process.response_time})",
                         [process.pid], 
                         {p.pid: "Executing" if p.pid == process.pid else "Waiting" if p.arrival_time <= self.current_time else "Not arrived" for p in self.processes})
            
            # Execute process completely
            self.gantt_chart.append((process.pid, self.current_time, 
                                   self.current_time + process.burst_time))
            
            self.current_time += process.burst_time
            process.completion_time = self.current_time
            process.turnaround_time = process.completion_time - process.arrival_time
            process.waiting_time = process.turnaround_time - process.burst_time
            
            self.add_step("COMPLETE", process.pid, 
                         f"P{process.pid} completes at time {self.current_time} (TAT: {process.turnaround_time}, WT: {process.waiting_time})",
                         [], 
                         {p.pid: "Completed" if p.pid == process.pid else "Waiting" if p.arrival_time <= self.current_time else "Not arrived" for p in self.processes})
        
        self.add_step("END", -1, "FCFS Algorithm completed")
        return self.gantt_chart.copy(), self.calculate_metrics()
    
    def sjf_non_preemptive(self) -> Tuple[List, Dict]:
        """Shortest Job First (Non-preemptive) scheduling with step tracking"""
        self.reset()
        self.add_step("INIT", -1, "Starting SJF Non-preemptive Algorithm")
        
        completed = []
        
        while len(completed) < len(self.processes):
            # Find available processes at current time
            available = [p for p in self.processes 
                        if p.arrival_time <= self.current_time and p not in completed]
            
            if not available:
                # No process available, advance time to next arrival
                next_arrival = min(p.arrival_time for p in self.processes if p not in completed)
                self.add_step("WAIT", -1, f"CPU idle from time {self.current_time} to {next_arrival} (waiting for next process arrival)")
                self.current_time = next_arrival
                continue
            
            # Select process with shortest burst time
            shortest = min(available, key=lambda p: p.burst_time)
            self.add_step("SELECT", shortest.pid, 
                         f"Selected P{shortest.pid} (burst time: {shortest.burst_time}) from available processes: {[f'P{p.pid}(BT:{p.burst_time})' for p in available]}")
            
            # Set start and response time
            if shortest.start_time == -1:
                shortest.start_time = self.current_time
                shortest.response_time = self.current_time - shortest.arrival_time
            
            self.add_step("START", shortest.pid, 
                         f"P{shortest.pid} starts execution at time {self.current_time} (Response time: {shortest.response_time})")
            
            # Execute completely
            self.gantt_chart.append((shortest.pid, self.current_time, 
                                   self.current_time + shortest.burst_time))
            
            self.current_time += shortest.burst_time
            shortest.completion_time = self.current_time
            shortest.turnaround_time = shortest.completion_time - shortest.arrival_time
            shortest.waiting_time = shortest.turnaround_time - shortest.burst_time
            completed.append(shortest)
            
            self.add_step("COMPLETE", shortest.pid, 
                         f"P{shortest.pid} completes at time {self.current_time} (TAT: {shortest.turnaround_time}, WT: {shortest.waiting_time})")
            
        self.add_step("END", -1, "SJF Non-preemptive Algorithm completed")
        return self.gantt_chart.copy(), self.calculate_metrics()
    
    def sjf_preemptive(self) -> Tuple[List, Dict]:
        """Shortest Job First (Preemptive/SRTF) scheduling with step tracking"""
        self.reset()
        self.add_step("INIT", -1, "Starting SJF Preemptive (SRTF) Algorithm")
        
        completed = []
        current_process = None
        
        # Find maximum time needed
        max_time = max(p.arrival_time + p.burst_time for p in self.processes) + 10
        
        for time in range(max_time):
            if len(completed) == len(self.processes):
                break
                
            # Add newly arrived processes
            arrived = [p for p in self.processes 
                      if p.arrival_time <= time and p not in completed and p.remaining_time > 0]
            
            if not arrived:
                continue
            
            # Find process with shortest remaining time
            shortest = min(arrived, key=lambda p: p.remaining_time)
            
            # Check for preemption
            if current_process != shortest:
                if current_process:
                    self.add_step("PREEMPT", current_process.pid, 
                                 f"P{current_process.pid} preempted at time {time} (remaining: {current_process.remaining_time})")
                
                current_process = shortest
                if shortest.start_time == -1:
                    shortest.start_time = time
                    shortest.response_time = time - shortest.arrival_time
                
                self.add_step("SELECT", shortest.pid, 
                             f"P{shortest.pid} selected at time {time} (remaining time: {shortest.remaining_time}) from: {[f'P{p.pid}(RT:{p.remaining_time})' for p in arrived]}")
            
            # Execute for 1 time unit
            if len(self.gantt_chart) == 0 or self.gantt_chart[-1][0] != current_process.pid:
                self.gantt_chart.append((current_process.pid, time, time + 1))
            else:
                # Extend current execution
                last_entry = list(self.gantt_chart[-1])
                last_entry[2] = time + 1
                self.gantt_chart[-1] = tuple(last_entry)
            
            current_process.remaining_time -= 1
            self.current_time = time + 1
            
            # Check if process completed
            if current_process.remaining_time == 0:
                current_process.completion_time = time + 1
                current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
                current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
                completed.append(current_process)
                
                self.add_step("COMPLETE", current_process.pid, 
                             f"P{current_process.pid} completes at time {time + 1} (TAT: {current_process.turnaround_time}, WT: {current_process.waiting_time})")
                current_process = None
        
        self.add_step("END", -1, "SJF Preemptive Algorithm completed")
        return self.gantt_chart.copy(), self.calculate_metrics()
    
    def round_robin(self, time_quantum: int = 2) -> Tuple[List, Dict]:
        """Round Robin scheduling with step tracking"""
        self.reset()
        self.add_step("INIT", -1, f"Starting Round Robin Algorithm (Time Quantum: {time_quantum})")
        
        ready_queue = deque()
        completed = []
        
        # Find maximum time needed
        max_time = max(p.arrival_time + p.burst_time for p in self.processes) + 20
        
        for time in range(max_time):
            if len(completed) == len(self.processes):
                break
                
            # Add newly arrived processes to ready queue
            for process in self.processes:
                if (process.arrival_time <= time and 
                    process not in ready_queue and 
                    process not in completed and
                    process.remaining_time > 0):
                    ready_queue.append(process)
                    self.add_step("ARRIVE", process.pid, 
                                 f"P{process.pid} arrives and joins ready queue at time {time}",
                                 [p.pid for p in ready_queue])
            
            if ready_queue:
                current_process = ready_queue.popleft()
                
                # Set start and response time
                if current_process.start_time == -1:
                    current_process.start_time = time
                    current_process.response_time = time - current_process.arrival_time
                
                self.add_step("SELECT", current_process.pid, 
                             f"P{current_process.pid} selected from ready queue at time {time} (remaining: {current_process.remaining_time})",
                             [p.pid for p in ready_queue])
                
                # Execute for time quantum or remaining time
                execution_time = min(time_quantum, current_process.remaining_time)
                self.gantt_chart.append((current_process.pid, time, time + execution_time))
                
                current_process.remaining_time -= execution_time
                time += execution_time - 1  # -1 because loop will increment
                self.current_time = time + 1
                
                # Check if process completed
                if current_process.remaining_time == 0:
                    current_process.completion_time = time + 1
                    current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
                    current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
                    completed.append(current_process)
                    
                    self.add_step("COMPLETE", current_process.pid, 
                                 f"P{current_process.pid} completes at time {time + 1} (TAT: {current_process.turnaround_time}, WT: {current_process.waiting_time})")
                else:
                    # Add back to queue if not completed
                    ready_queue.append(current_process)
                    self.add_step("QUANTUM_EXPIRE", current_process.pid, 
                                 f"P{current_process.pid} quantum expires, rejoins queue (remaining: {current_process.remaining_time})",
                                 [p.pid for p in ready_queue])
        
        self.add_step("END", -1, "Round Robin Algorithm completed")
        return self.gantt_chart.copy(), self.calculate_metrics()
    
    def priority_non_preemptive(self) -> Tuple[List, Dict]:
        """Priority scheduling (Non-preemptive) with step tracking"""
        self.reset()
        self.add_step("INIT", -1, "Starting Priority Non-preemptive Algorithm (lower number = higher priority)")
        
        completed = []
        
        while len(completed) < len(self.processes):
            # Find available processes at current time
            available = [p for p in self.processes 
                        if p.arrival_time <= self.current_time and p not in completed]
            
            if not available:
                # No process available, advance time to next arrival
                next_arrival = min(p.arrival_time for p in self.processes if p not in completed)
                self.add_step("WAIT", -1, f"CPU idle from time {self.current_time} to {next_arrival}")
                self.current_time = next_arrival
                continue
            
            # Select process with highest priority (lowest number)
            highest_priority = min(available, key=lambda p: p.priority)
            self.add_step("SELECT", highest_priority.pid, 
                         f"Selected P{highest_priority.pid} (priority: {highest_priority.priority}) from available: {[f'P{p.pid}(Pr:{p.priority})' for p in available]}")
            
            # Set start and response time
            if highest_priority.start_time == -1:
                highest_priority.start_time = self.current_time
                highest_priority.response_time = self.current_time - highest_priority.arrival_time
            
            self.add_step("START", highest_priority.pid, 
                         f"P{highest_priority.pid} starts execution at time {self.current_time} (Response time: {highest_priority.response_time})")
            
            # Execute completely
            self.gantt_chart.append((highest_priority.pid, self.current_time, 
                                   self.current_time + highest_priority.burst_time))
            
            self.current_time += highest_priority.burst_time
            highest_priority.completion_time = self.current_time
            highest_priority.turnaround_time = highest_priority.completion_time - highest_priority.arrival_time
            highest_priority.waiting_time = highest_priority.turnaround_time - highest_priority.burst_time
            completed.append(highest_priority)
            
            self.add_step("COMPLETE", highest_priority.pid, 
                         f"P{highest_priority.pid} completes at time {self.current_time} (TAT: {highest_priority.turnaround_time}, WT: {highest_priority.waiting_time})")
            
        self.add_step("END", -1, "Priority Non-preemptive Algorithm completed")
        return self.gantt_chart.copy(), self.calculate_metrics()
    
    def priority_preemptive(self) -> Tuple[List, Dict]:
        """Priority scheduling (Preemptive) with step tracking"""
        self.reset()
        self.add_step("INIT", -1, "Starting Priority Preemptive Algorithm (lower number = higher priority)")
        
        completed = []
        current_process = None
        
        # Find maximum time needed
        max_time = max(p.arrival_time + p.burst_time for p in self.processes) + 10
        
        for time in range(max_time):
            if len(completed) == len(self.processes):
                break
                
            # Add newly arrived processes
            arrived = [p for p in self.processes 
                      if p.arrival_time <= time and p not in completed and p.remaining_time > 0]
            
            if not arrived:
                continue
            
            # Find process with highest priority (lowest number)
            highest_priority = min(arrived, key=lambda p: p.priority)
            
            # Check for preemption
            if current_process != highest_priority:
                if current_process:
                    self.add_step("PREEMPT", current_process.pid, 
                                 f"P{current_process.pid} preempted by higher priority process at time {time}")
                
                current_process = highest_priority
                if highest_priority.start_time == -1:
                    highest_priority.start_time = time
                    highest_priority.response_time = time - highest_priority.arrival_time
                
                self.add_step("SELECT", highest_priority.pid, 
                             f"P{highest_priority.pid} selected at time {time} (priority: {highest_priority.priority}) from: {[f'P{p.pid}(Pr:{p.priority})' for p in arrived]}")
            
            # Execute for 1 time unit
            if len(self.gantt_chart) == 0 or self.gantt_chart[-1][0] != current_process.pid:
                self.gantt_chart.append((current_process.pid, time, time + 1))
            else:
                # Extend current execution
                last_entry = list(self.gantt_chart[-1])
                last_entry[2] = time + 1
                self.gantt_chart[-1] = tuple(last_entry)
            
            current_process.remaining_time -= 1
            self.current_time = time + 1
            
            # Check if process completed
            if current_process.remaining_time == 0:
                current_process.completion_time = time + 1
                current_process.turnaround_time = current_process.completion_time - current_process.arrival_time
                current_process.waiting_time = current_process.turnaround_time - current_process.burst_time
                completed.append(current_process)
                
                self.add_step("COMPLETE", current_process.pid, 
                             f"P{current_process.pid} completes at time {time + 1} (TAT: {current_process.turnaround_time}, WT: {current_process.waiting_time})")
                current_process = None
        
        self.add_step("END", -1, "Priority Preemptive Algorithm completed")
        return self.gantt_chart.copy(), self.calculate_metrics()

class DeadlockManager:
    """Implements Banker's Algorithm for deadlock avoidance"""
    
    def __init__(self, resources: List[Resource]):
        self.resources = resources
        self.processes = []
        self.allocation_matrix = []
        self.max_matrix = []
        self.need_matrix = []
        self.available_vector = []
        self.safe_sequence = []
        
    def add_process_resource_info(self, pid: int, allocation: List[int], max_need: List[int]):
        """Add process resource allocation and maximum need information"""
        if len(allocation) != len(self.resources) or len(max_need) != len(self.resources):
            raise ValueError("Allocation and max_need must match number of resources")
        
        self.processes.append(pid)
        self.allocation_matrix.append(allocation.copy())
        self.max_matrix.append(max_need.copy())
        
        # Calculate need matrix (Need = Max - Allocation)
        need = [max_need[i] - allocation[i] for i in range(len(allocation))]
        self.need_matrix.append(need)
        
        # Update available vector
        self.available_vector = [r.available_instances for r in self.resources]
    
    def is_safe_state(self) -> Tuple[bool, List[int]]:
        """Check if current state is safe using Banker's Algorithm"""
        if not self.processes:
            return True, []
        
        work = self.available_vector.copy()
        finish = [False] * len(self.processes)
        safe_sequence = []
        
        while len(safe_sequence) < len(self.processes):
            found = False
            
            for i in range(len(self.processes)):
                if not finish[i]:
                    # Check if process can be satisfied
                    can_allocate = all(self.need_matrix[i][j] <= work[j] 
                                     for j in range(len(self.resources)))
                    
                    if can_allocate:
                        # Process can complete, add its resources back to work
                        for j in range(len(self.resources)):
                            work[j] += self.allocation_matrix[i][j]
                        
                        finish[i] = True
                        safe_sequence.append(self.processes[i])
                        found = True
                        break
            
            if not found:
                # No process can proceed, unsafe state
                return False, []
        
        self.safe_sequence = safe_sequence
        return True, safe_sequence
    
    def request_resources(self, pid: int, request: List[int]) -> Tuple[bool, str]:
        """Process resource request using Banker's Algorithm"""
        try:
            process_index = self.processes.index(pid)
        except ValueError:
            return False, f"Process {pid} not found"
        
        # Check if request exceeds need
        for i in range(len(request)):
            if request[i] > self.need_matrix[process_index][i]:
                return False, f"Request exceeds maximum need for resource {i}"
        
        # Check if request exceeds available
        for i in range(len(request)):
            if request[i] > self.available_vector[i]:
                return False, f"Request exceeds available resources for resource {i}"
        
        # Tentatively allocate resources
        old_allocation = self.allocation_matrix[process_index].copy()
        old_available = self.available_vector.copy()
        old_need = self.need_matrix[process_index].copy()
        
        for i in range(len(request)):
            self.allocation_matrix[process_index][i] += request[i]
            self.available_vector[i] -= request[i]
            self.need_matrix[process_index][i] -= request[i]
        
        # Check if resulting state is safe
        is_safe, sequence = self.is_safe_state()
        
        if is_safe:
            return True, f"Request granted. Safe sequence: {sequence}"
        else:
            # Rollback allocation
            self.allocation_matrix[process_index] = old_allocation
            self.available_vector = old_available
            self.need_matrix[process_index] = old_need
            return False, "Request denied: would lead to unsafe state"

class OSSimulationGUI:
    def __init__(self, master):
        self.master = master
        # Your other initialization code
        self.setup_gui()
    
    def setup_gui(self):
        # Your existing GUI setup code
        self.setup_cpu_tab()
    
    def setup_cpu_tab(self):
        # Your existing CPU tab setup code
        # Make sure you have something like:
        compare_button = ttk.Button(cpu_frame, text="Compare All", command=self.run_all_algorithms)
        compare_button.pack()
    
    def run_all_algorithms(self):
        """Run and compare all algorithms with results in a table"""
        if not hasattr(self, 'scheduler') or not self.scheduler.processes:
            messagebox.showwarning("Warning", "No processes added")
            return
        
        # Get quantum time if available, default to 2
        quantum = 2
        if hasattr(self, 'quantum_entry') and self.quantum_entry.get():
            try:
                quantum = int(self.quantum_entry.get())
            except ValueError:
                messagebox.showwarning("Warning", "Invalid quantum time, using default (2)")
                quantum = 2
        
        algorithms = [
            ("FCFS", lambda: self.scheduler.fcfs()),
            ("SJF (Non-preemptive)", lambda: self.scheduler.sjf_non_preemptive()),
            ("SJF (Preemptive)", lambda: self.scheduler.sjf_preemptive()),
            ("Round Robin", lambda: self.scheduler.round_robin(quantum)),
            ("Priority (Non-preemptive)", lambda: self.scheduler.priority_non_preemptive()),
            ("Priority (Preemptive)", lambda: self.scheduler.priority_preemptive())
        ]
        
        # Collect all results first
        results = []
        for name, func in algorithms:
            try:
                _, metrics = func()
                results.append((name, metrics))
            except Exception as e:
                results.append((name, {"error": str(e)}))
        
        # Create table header
        table = "+----------------------+------------+------------+------------+------------+------------+\n"
        table += "| Algorithm            | Avg WT     | Avg TAT    | Avg RT     | CPU Util % | Throughput |\n"
        table += "+======================+============+============+============+============+============+\n"
        
        # Add each algorithm's results to the table
        for name, metrics in results:
            if "error" in metrics:
                table += f"| {name:<20} | {'Error:':<10} | {metrics['error'][:10]:<10} | {'':<10} | {'':<10} | {'':<10} |\n"
            else:
                table += f"| {name:<20} | {metrics['avg_waiting_time']:>10.2f} | {metrics['avg_turnaround_time']:>10.2f} | "
                table += f"{metrics['avg_response_time']:>10.2f} | {metrics['cpu_utilization']:>10.2f} | {metrics['throughput']:>10.2f} |\n"
        
        table += "+----------------------+------------+------------+------------+------------+------------+\n"
        
        # Add some analysis
        analysis = "\n=== ANALYSIS ===\n"
        
        # Find best algorithm for each metric
        if all("error" not in m for _, m in results):
            best_wt = min(results, key=lambda x: x[1]['avg_waiting_time'])
            best_tat = min(results, key=lambda x: x[1]['avg_turnaround_time'])
            best_rt = min(results, key=lambda x: x[1]['avg_response_time'])
            best_cpu = max(results, key=lambda x: x[1]['cpu_utilization'])
            best_throughput = max(results, key=lambda x: x[1]['throughput'])
            
            analysis += f"\nBest Average Waiting Time: {best_wt[0]} ({best_wt[1]['avg_waiting_time']:.2f})\n"
            analysis += f"Best Average Turnaround Time: {best_tat[0]} ({best_tat[1]['avg_turnaround_time']:.2f})\n"
            analysis += f"Best Average Response Time: {best_rt[0]} ({best_rt[1]['avg_response_time']:.2f})\n"
            analysis += f"Best CPU Utilization: {best_cpu[0]} ({best_cpu[1]['cpu_utilization']:.2f}%)\n"
            analysis += f"Best Throughput: {best_throughput[0]} ({best_throughput[1]['throughput']:.2f})\n"
        
        # Display results
        if not hasattr(self, 'results_text'):
            self.results_text = scrolledtext.ScrolledText(self.results_frame, height=30)
            self.results_text.pack(fill='both', expand=True)
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "=== ALGORITHM COMPARISON ===\n\n")
        self.results_text.insert(tk.END, table)
        self.results_text.insert(tk.END, analysis)
        
        if hasattr(self, 'notebook'):
            self.notebook.select(self.results_frame)

def main():
    root = tk.Tk()
    app = OSSimulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()