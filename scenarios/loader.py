from .task1_scenarios import TASK1_SCENARIOS
from .task2_scenarios import TASK2_SCENARIOS
from .task3_scenarios import TASK3_SCENARIOS
from .task4_scenarios import TASK4_SCENARIOS
from .fleet_scenarios import FLEET_SCENARIOS, FLEET_OVERSIGHT_SCENARIOS
import random

class ScenarioLoader:
    @staticmethod
    def get_task1(scenario_id: str = None):
        if scenario_id and scenario_id in TASK1_SCENARIOS:
            return TASK1_SCENARIOS[scenario_id]
        return TASK1_SCENARIOS[random.choice(list(TASK1_SCENARIOS.keys()))]

    @staticmethod
    def get_task2(scenario_id: str = None):
        if scenario_id and scenario_id in TASK2_SCENARIOS:
            return TASK2_SCENARIOS[scenario_id]
        return TASK2_SCENARIOS[random.choice(list(TASK2_SCENARIOS.keys()))]

    @staticmethod
    def get_task3(scenario_id: str = None):
        if scenario_id and scenario_id in TASK3_SCENARIOS:
            return TASK3_SCENARIOS[scenario_id]
        return TASK3_SCENARIOS[random.choice(list(TASK3_SCENARIOS.keys()))]

    @staticmethod
    def get_task4(scenario_id: str = None):
        if scenario_id and scenario_id in TASK4_SCENARIOS:
            return TASK4_SCENARIOS[scenario_id]
        return TASK4_SCENARIOS[random.choice(list(TASK4_SCENARIOS.keys()))]

    @staticmethod
    def get_fleet(scenario_id: str = None):
        if scenario_id and scenario_id in FLEET_SCENARIOS:
            return FLEET_SCENARIOS[scenario_id]
        return FLEET_SCENARIOS[random.choice(list(FLEET_SCENARIOS.keys()))]

    @staticmethod
    def get_fleet_oversight(scenario_id: str = None):
        if scenario_id and scenario_id in FLEET_OVERSIGHT_SCENARIOS:
            return FLEET_OVERSIGHT_SCENARIOS[scenario_id]
        return FLEET_OVERSIGHT_SCENARIOS[random.choice(list(FLEET_OVERSIGHT_SCENARIOS.keys()))]

    @staticmethod
    def list_scenarios():
        return {
            "precision_assignment": list(TASK1_SCENARIOS.keys()),
            "instability_detection": list(TASK2_SCENARIOS.keys()),
            "multi_objective_optimization": list(TASK3_SCENARIOS.keys()),
            "precision_transfer": list(TASK4_SCENARIOS.keys()),
            "fleet_precision": list(FLEET_SCENARIOS.keys()),
            "fleet_oversight": list(FLEET_OVERSIGHT_SCENARIOS.keys()),
            "fleet_resource": list(FLEET_SCENARIOS.keys()),
            "fleet_recovery": [k for k, v in FLEET_OVERSIGHT_SCENARIOS.items()
                               if v["ground_truth"]["crashing_model"] is not None],
        }
