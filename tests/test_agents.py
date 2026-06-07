import os
import pytest
from src.sustainable_campus.config import default_config
from src.sustainable_campus.agents.communication import CampusEnvironment, Message
from src.sustainable_campus.agents.classroom_agent import ClassroomAgent
from src.sustainable_campus.agents.laboratory_agent import LaboratoryAgent
from src.sustainable_campus.agents.hostel_agent import HostelAgent
from src.sustainable_campus.agents.energy_manager_agent import EnergyManagerAgent

@pytest.fixture
def env_setup():
    env = CampusEnvironment(
        total_capacity_kwh=default_config.grid.total_capacity_kwh,
        weights=default_config.weights
    )
    return env

def test_classroom_agent(env_setup):
    env = env_setup
    agent = ClassroomAgent("classroom", env, default_config.classroom)
    
    # Setup env state
    env.set_step_state(
        step=0, hour=10, day=0, 
        occupancy={"classroom": 150}, 
        temperature=24.0, is_holiday=False, is_exam_period=False
    )
    
    request = agent.make_request()
    assert request > 0.0
    assert len(agent.request_history) == 1
    
    satisfaction = agent.update_state(request * 0.8)
    assert satisfaction < 1.0
    assert len(agent.allocation_history) == 1
    assert agent.utilization == 150 / default_config.classroom.base_occupancy

def test_laboratory_agent(env_setup):
    env = env_setup
    agent = LaboratoryAgent("laboratory", env, default_config.laboratory)
    
    env.set_step_state(
        step=0, hour=12, day=1, 
        occupancy={"laboratory": 30}, 
        temperature=22.0, is_holiday=False, is_exam_period=True
    )
    
    request = agent.make_request()
    assert request > 0.0
    
    forecast = agent.forecast_future_usage()
    assert forecast > 0.0
    
    satisfaction = agent.update_state(request)
    assert satisfaction == 1.0

def test_hostel_agent(env_setup):
    env = env_setup
    agent = HostelAgent("hostel", env, default_config.hostel)
    
    env.set_step_state(
        step=0, hour=20, day=2, 
        occupancy={"hostel": 900}, 
        temperature=18.0, is_holiday=False, is_exam_period=False
    )
    
    request = agent.make_request()
    assert request > 0.0
    
    # Over-allocating to test efficiency penalty
    agent.update_state(request * 1.5)
    assert agent.efficiency_score < 1.0
    
    # Under-allocating to test efficiency behavior
    agent.update_state(request * 0.8)
    assert agent.efficiency_score <= 1.0

def test_energy_manager(env_setup):
    env = env_setup
    manager = EnergyManagerAgent("energy_manager", env, default_config.grid, default_config.weights)
    
    classroom = ClassroomAgent("classroom", env, default_config.classroom)
    hostel = HostelAgent("hostel", env, default_config.hostel)
    
    env.set_step_state(
        step=0, hour=10, day=0, 
        occupancy={"classroom": 100, "hostel": 500}, 
        temperature=22.0, is_holiday=False, is_exam_period=False
    )
    
    # Simulate requests
    req_c = classroom.make_request()
    req_h = hostel.make_request()
    
    manager.receive_requests()
    assert len(manager.current_requests) == 2
    
    # Test Proportional Allocation
    allocations = manager.allocate_resources(policy="equal_proportion")
    assert sum(allocations.values()) <= default_config.grid.total_capacity_kwh
    
    # Update agent states
    classroom.update_state(allocations["classroom"])
    hostel.update_state(allocations["hostel"])
    
    manager.receive_requests() # Read confirmations
    metrics = manager.collect_confirms()
    
    assert "average_satisfaction" in metrics
    assert "sustainability_score" in metrics
    assert metrics["total_energy_consumed"] == sum(allocations.values())
