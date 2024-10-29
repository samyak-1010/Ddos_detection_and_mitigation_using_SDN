import heapq
import random
import time

class Request:
    def __init__(self, arrival_time, processing_time, priority):
        self.arrival_time = arrival_time
        self.processing_time = processing_time
        self.priority = priority  # Lower number = higher priority

    def __lt__(self, other):
        return self.priority < other.priority  # Priority comparison

class QueuingSystem:
    def __init__(self):
        self.priority_queue = []  # Min-heap for priority queue
        self.current_time = 0     # Simulation time

    def add_request(self, request):
        heapq.heappush(self.priority_queue, request)

    def process_requests(self):
        while self.priority_queue:
            # Process the highest priority request
            request = heapq.heappop(self.priority_queue)
            # Wait until the request's arrival time
            self.current_time = max(self.current_time, request.arrival_time)
            print(f"Processing request with priority {request.priority} at time {self.current_time}")
            time.sleep(request.processing_time)  # Simulate processing time
            self.current_time += request.processing_time
            print(f"Finished processing request. Time now: {self.current_time}")

    def simulate(self, num_requests):
        for i in range(num_requests):
            arrival_time = self.current_time + random.randint(1, 5)  # Random arrival time
            processing_time = random.uniform(0.5, 2)  # Random processing time
            priority = random.choice([1, 2, 3])  # Lower number is higher priority
            request = Request(arrival_time, processing_time, priority)
            self.add_request(request)
            print(f"Added request with priority {priority} arriving at {arrival_time}, processing time {processing_time:.2f}")

        # Sort requests by arrival time before processing
        self.priority_queue.sort(key=lambda req: req.arrival_time)
        self.process_requests()

# Example Usage
if __name__ == "__main__":
    queuing_system = QueuingSystem()
    num_requests = 10  # Number of requests to simulate
    queuing_system.simulate(num_requests)
