# Modified by Aslan Satary Dizaji, Copyright (c) 2023.

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from modified_ai_economist_wt.foundation.base.base_component import BaseComponent, component_registry
from copy import deepcopy

import pdb


@component_registry.add
class teaching_and_build_component(BaseComponent):
    """Mobile agents are divided to two categories, teachers and students. Teacher mobile agents 
       know how to build houses, but they are not able to do that. On the other hand, student mobile 
       agents don't know how to build houses, but if someone teaches them, they are able to do that.
       This class allows teacher mobile agents to teach (using communication with misalignment) 
       student mobile agents to build two types of houses (using complementary resources 
       ([Wood, Stone] and [Iron, Soil])) and earn income together.

    Can be configured to include heterogeneous building skill/labor where agents earn different levels of 
    income or tolerate different levels of labor when building houses. 

    Args:
        payment (int): Default amount of coin agents earn from building.
            Must be >= 0. Default is 10.
        payment_max_skill_multiplier (int array): Maximum skill multiplier that an agent
            can sample. Must be >= 1. Default is 1.
        skill_dist (str): Distribution type for sampling skills. Default ("none")
            gives all agents identical skill equal to a multiplier of 1. "pareto" and
            "lognormal" sample skills from the associated distributions.
        build_labor (float): Labor cost associated with building a house.
            Must be >= 0. Default is 10.
    """

    name = "TeachBuild"
    component_type = "TeachBuild"
    required_entities = ["Wood", "Stone", "Iron", "Soil", "Coin", "RedHouse", "BlueHouse", "Labor"]
    agent_subclasses = ["BasicMobileAgent"]
    
    def __init__(
        self,
        *base_component_args,
        payment=10,
        payment_max_skill_multiplier=np.array([10, 10]),
        skill_dist="pareto",
        build_labor=np.array([20.0, 20.0]),
        **base_component_kwargs
    ):
        
        super().__init__(*base_component_args, **base_component_kwargs)

        self.payment = int(payment)
        assert self.payment >= 0

        self.payment_max_skill_multiplier = payment_max_skill_multiplier
        assert np.all(self.payment_max_skill_multiplier >= 1)

        self.resource_cost = {"Wood": 1, "Stone": 1, "Iron": 1, "Soil": 1}

        self.build_labor = build_labor
        assert np.all(self.build_labor >= 0.0)

        self.skill_dist = skill_dist.lower()
        assert self.skill_dist in ["none", "pareto", "lognormal"]

        self.sampled_skills_red_house_together = {}
        self.sampled_skills_blue_house_together = {}

        self.builds = []

    def agent_can_build(self, agent):
        """Return True if agent can actually build in its current location."""
        
        # See if the agent has the resources necessary to complete the action
        for resource, cost in self.resource_cost.items():
            if agent.state["inventory"][resource] < cost:
                return False

        # Do nothing if this spot is already occupied by a landmark or resource
        if self.world.location_resources(*agent.loc):
            return False
        if self.world.location_landmarks(*agent.loc):
            return False
        # If we made it here, the agent can build.
        return True

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        Add three actions (building three different houses) for mobile agents.
        """
        
        # This component adds three actions that mobile agents can take: building three different houses.
        if agent_cls_name == "BasicMobileAgent":
            return 2

        return None

    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        For mobile agents, add state fields for building skill.
        """
        
        if agent_cls_name not in self.agent_subclasses:
            return {}
        if agent_cls_name == "BasicMobileAgent":
            return {"build_payment_red_house_together": float(self.payment), "build_payment_blue_house_together": float(self.payment),
                    "build_skill_red_house_together": 1, "build_skill_blue_house_together": 1}
        
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.

        Convert stone+wood+iron+soil to house+coin for agents that choose to build and can.
        """
        
        world = self.world
        build = []
        agents_counted_action_1 = []
        agents_counted_action_2 = []

        # Apply any building actions taken by the mobile agents
        for agent1 in world.get_random_order_agents():

            action1 = agent1.get_component_action(self.name)

            # This component doesn't apply to this agent!
            if action1 is None:
                continue

            # NO-OP!
            if action1 == 0:
                pass
            
            # Build a red house together if you can!
            elif action1 == 1 and agent1.state["endogenous"]["Teaching"][0] == "teacher":
                agents_counted_action_1.append(int(agent1.idx))

                # Apply any building actions taken by the mobile agents
                for agent2 in world.get_random_order_agents():

                    action2 = agent2.get_component_action(self.name)

                    if action2 == 1 and int(agent2.idx) not in agents_counted_action_1 and agent2.state["endogenous"]["Teaching"][0] == "student":

                        agents_counted_action_1.append(int(agent2.idx))
                
                        self.resource_cost = {"Wood": 1, "Stone": 1}

                        if agent1.state["endogenous"]["Teaching"][1][0] == agent2.state["endogenous"]["Teaching"][1][0]:
                            if agent1.state["endogenous"]["Teaching"][1][1] == agent2.state["endogenous"]["Teaching"][1][1]:

                                if self.agent_can_build(agent2):
                                    # Remove the resources
                                    agent2.state["inventory"]["Wood"] -= 1
                                    agent2.state["inventory"]["Stone"] -= 1

                                    # Place two houses where the agent1 and agent2 are standing
                                    loc_r, loc_c = agent1.loc
                                    world.create_landmark("RedHouse", loc_r, loc_c, agent1.idx)

                                    loc_r, loc_c = agent2.loc
                                    world.create_landmark("RedHouse", loc_r, loc_c, agent2.idx)

                                    # Receive payment for the house
                                    agent1.state["inventory"]["Coin"] += agent1.state["build_payment_red_house_together"]
                                    agent2.state["inventory"]["Coin"] += agent2.state["build_payment_red_house_together"]

                                    # Incur the labor cost for building
                                    agent1.state["endogenous"]["Labor"] += self.build_labor[0]
                                    agent2.state["endogenous"]["Labor"] += self.build_labor[0]

                                    build.append(
                                        {
                                            "builder": agent1.idx,
                                            "type": "red_teacher",
                                            "loc": np.array(agent1.loc),
                                            "income": float(agent1.state["build_payment_red_house_together"]),
                                        }
                                    )

                                    build.append(
                                        {
                                            "builder": agent2.idx,
                                            "type": "red_student",
                                            "loc": np.array(agent2.loc),
                                            "income": float(agent2.state["build_payment_red_house_together"]),
                                        }
                                    )

                            elif agent1.state["endogenous"]["Teaching"][1][1] != agent2.state["endogenous"]["Teaching"][1][1]:

                                for i in range(len(agent2.state["endogenous"]["Teaching"][1])):
                                    if agent2.state["endogenous"]["Teaching"][1][i] == agent1.state["endogenous"]["Teaching"][1][1]:
                                        temp = deepcopy(agent2.state["endogenous"]["Teaching"][1][1])
                                        agent2.state["endogenous"]["Teaching"][1][1] = deepcopy(agent1.state["endogenous"]["Teaching"][1][1])
                                        agent2.state["endogenous"]["Teaching"][1][i] = deepcopy(temp)
                                
                                # Receive small payment
                                agent1.state["inventory"]["Coin"] += 2
                                agent2.state["inventory"]["Coin"] += 2

                                # Incur small labor cost
                                agent1.state["endogenous"]["Labor"] += 1
                                agent2.state["endogenous"]["Labor"] += 1

                        elif agent1.state["endogenous"]["Teaching"][1][0] != agent2.state["endogenous"]["Teaching"][1][0]:

                            for i in range(len(agent2.state["endogenous"]["Teaching"][1])):
                                if agent2.state["endogenous"]["Teaching"][1][i] == agent1.state["endogenous"]["Teaching"][1][0]:
                                    temp = deepcopy(agent2.state["endogenous"]["Teaching"][1][0])
                                    agent2.state["endogenous"]["Teaching"][1][0] = deepcopy(agent1.state["endogenous"]["Teaching"][1][0])
                                    agent2.state["endogenous"]["Teaching"][1][i] = deepcopy(temp)

                            # Receive small payment
                            agent1.state["inventory"]["Coin"] += 2
                            agent2.state["inventory"]["Coin"] += 2

                            # Incur small labor cost
                            agent1.state["endogenous"]["Labor"] += 1
                            agent2.state["endogenous"]["Labor"] += 1

            # Build a blue house together if you can!
            elif action1 == 2 and agent1.state["endogenous"]["Teaching"][0] == "teacher":
                agents_counted_action_2.append(int(agent1.idx))

                # Apply any building actions taken by the mobile agents
                for agent2 in world.get_random_order_agents():

                    action2 = agent2.get_component_action(self.name)

                    if action2 == 2 and int(agent2.idx) not in agents_counted_action_2 and agent2.state["endogenous"]["Teaching"][0] == "student":

                        agents_counted_action_2.append(int(agent2.idx))
                
                        self.resource_cost = {"Iron": 1, "Soil": 1}

                        if agent1.state["endogenous"]["Teaching"][1][2] == agent2.state["endogenous"]["Teaching"][1][2]:
                            if agent1.state["endogenous"]["Teaching"][1][3] == agent2.state["endogenous"]["Teaching"][1][3]:

                                if self.agent_can_build(agent2):
                                    # Remove the resources
                                    agent2.state["inventory"]["Iron"] -= 1
                                    agent2.state["inventory"]["Soil"] -= 1

                                    # Place two houses where the agent1 and agent2 are standing
                                    loc_r, loc_c = agent1.loc
                                    world.create_landmark("BlueHouse", loc_r, loc_c, agent1.idx)

                                    loc_r, loc_c = agent2.loc
                                    world.create_landmark("BlueHouse", loc_r, loc_c, agent2.idx)

                                    # Receive payment for the house
                                    agent1.state["inventory"]["Coin"] += agent1.state["build_payment_blue_house_together"]
                                    agent2.state["inventory"]["Coin"] += agent2.state["build_payment_blue_house_together"]

                                    # Incur the labor cost for building
                                    agent1.state["endogenous"]["Labor"] += self.build_labor[1]
                                    agent2.state["endogenous"]["Labor"] += self.build_labor[1]

                                    build.append(
                                        {
                                            "builder": agent1.idx,
                                            "type": "blue_teacher",
                                            "loc": np.array(agent1.loc),
                                            "income": float(agent1.state["build_payment_blue_house_together"]),
                                        }
                                    )

                                    build.append(
                                        {
                                            "builder": agent2.idx,
                                            "type": "blue_student",
                                            "loc": np.array(agent2.loc),
                                            "income": float(agent2.state["build_payment_blue_house_together"]),
                                        }
                                    )

                            elif agent1.state["endogenous"]["Teaching"][1][3] != agent2.state["endogenous"]["Teaching"][1][3]:

                                for i in range(len(agent2.state["endogenous"]["Teaching"][1])):
                                    if agent2.state["endogenous"]["Teaching"][1][i] == agent1.state["endogenous"]["Teaching"][1][3]:
                                        temp = deepcopy(agent2.state["endogenous"]["Teaching"][1][3])
                                        agent2.state["endogenous"]["Teaching"][1][3] = deepcopy(agent1.state["endogenous"]["Teaching"][1][3])
                                        agent2.state["endogenous"]["Teaching"][1][i] = deepcopy(temp)

                                # Receive small payment
                                agent1.state["inventory"]["Coin"] += 2
                                agent2.state["inventory"]["Coin"] += 2

                                # Incur small labor cost
                                agent1.state["endogenous"]["Labor"] += 1
                                agent2.state["endogenous"]["Labor"] += 1

                        elif agent1.state["endogenous"]["Teaching"][1][2] != agent2.state["endogenous"]["Teaching"][1][2]:

                            for i in range(len(agent2.state["endogenous"]["Teaching"][1])):
                                if agent2.state["endogenous"]["Teaching"][1][i] == agent1.state["endogenous"]["Teaching"][1][2]:
                                    temp = deepcopy(agent2.state["endogenous"]["Teaching"][1][2])
                                    agent2.state["endogenous"]["Teaching"][1][2] = deepcopy(agent1.state["endogenous"]["Teaching"][1][2])
                                    agent2.state["endogenous"]["Teaching"][1][i] = deepcopy(temp)

                            # Receive small payment
                            agent1.state["inventory"]["Coin"] += 2
                            agent2.state["inventory"]["Coin"] += 2

                            # Incur small labor cost
                            agent1.state["endogenous"]["Labor"] += 1
                            agent2.state["endogenous"]["Labor"] += 1
                                            
        self.builds.append(build)

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Here, agents observe their build skill. The planner does not observe anything
        from this component.
        """

        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                "build_payment_red_house_together": agent.state["build_payment_red_house_together"] / self.payment,
                "build_payment_blue_house_together": agent.state["build_payment_blue_house_together"] / self.payment,
                "build_skill_red_house_together": agent.state["build_skill_red_house_together"],
                "build_skill_blue_house_together": agent.state["build_skill_blue_house_together"],
            }

        return obs_dict

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.

        Prevent building only if a landmark already occupies the agent's location.
        """

        masks = {}
        # Mobile agents' build action is masked if they cannot build with their current location and/or endowment
        for agent in self.world.agents:
            masks[agent.idx] = np.array([int(self.agent_can_build(agent)), int(self.agent_can_build(agent))])

        return masks

    # For non-required customization
    # ------------------------------

    def get_metrics(self):
        """
        Metrics that capture what happened through this component.

        Returns:
            metrics (dict): A dictionary of {"metric_name": metric_value},
                where metric_value is a scalar.
        """
        
        world = self.world

        build_stats = {a.idx: {"n_builds": 0} for a in world.agents}
        for builds in self.builds:
            for build in builds:
                idx = build["builder"]
                build_stats[idx]["n_builds"] += 1

        out_dict = {}
        for a in world.agents:
            for k, v in build_stats[a.idx].items():
                out_dict["{}/{}".format(a.idx, k)] = v

        num_houses = np.sum(world.maps.get("RedHouse") > 0) + np.sum(world.maps.get("BlueHouse") > 0)
        out_dict["total_builds"] = num_houses

        return out_dict

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Re-sample agents' building skills.
        """
        world = self.world

        self.sampled_skills_red_house_together = {agent.idx: 1 for agent in world.agents}
        self.sampled_skills_blue_house_together = {agent.idx: 1 for agent in world.agents}

        PMSM_red_together = self.payment_max_skill_multiplier[0]
        PMSM_blue_together = self.payment_max_skill_multiplier[1]

        for agent in world.agents:
            if self.skill_dist == "none":
                sampled_skill_red_house_together = 1
                sampled_skill_blue_house_together = 1
            
                pay_rate_red_together = 1
                pay_rate_blue_together = 1
            elif self.skill_dist == "pareto":
                sampled_skill_red_house_together = np.random.pareto(4)
                sampled_skill_blue_house_together = np.random.pareto(4)
                
                pay_rate_red_together = np.minimum(PMSM_red_together, (PMSM_red_together - 1) * sampled_skill_red_house_together + 1)
                pay_rate_blue_together = np.minimum(PMSM_blue_together, (PMSM_blue_together - 1) * sampled_skill_blue_house_together + 1)
            elif self.skill_dist == "lognormal":
                sampled_skill_red_house_together = np.random.lognormal(-1, 0.5)
                sampled_skill_blue_house_together = np.random.lognormal(-1, 0.5)
                
                pay_rate_red_together = np.minimum(PMSM_red_together, (PMSM_red_together - 1) * sampled_skill_red_house_together + 1)
                pay_rate_blue_together = np.minimum(PMSM_blue_together, (PMSM_blue_together - 1) * sampled_skill_blue_house_together + 1)
            else:
                raise NotImplementedError

            agent.state["build_payment_red_house_together"] = float(pay_rate_red_together * self.payment)
            agent.state["build_skill_red_house_together"] = float(sampled_skill_red_house_together)
            
            agent.state["build_payment_blue_house_together"] = float(pay_rate_blue_together * self.payment)
            agent.state["build_skill_blue_house_together"] = float(sampled_skill_blue_house_together)

            self.sampled_skills_red_house_together[agent.idx] = sampled_skill_red_house_together
            self.sampled_skills_blue_house_together[agent.idx] = sampled_skill_blue_house_together

        self.builds = []

    def get_dense_log(self):
        """
        Log builds.

        Returns:
            builds (list): A list of build events. Each entry corresponds to a single
                timestep and contains a description of any builds that occurred on
                that timestep.
        """
        
        return self.builds