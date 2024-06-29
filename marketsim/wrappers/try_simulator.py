from marketsim.simulator.sampled_arrival_simulator import SimulatorSampledArrival

surpluses = []

for _ in range(10):
    sim = SimulatorSampledArrival(num_background_agents=25,
                                  sim_time=12000,
                                  lam=5e-4,
                                  mean=1e5,
                                  r=0.05,
                                  shock_var=5e6,
                                  q_max=10,
                                  pv_var=5e6,
                                  shade=[250,500])
    sim.run()
    fundamental_val = sim.markets[0].get_final_fundamental()
    values = []
    for agent_id in sim.agents:
        agent = sim.agents[agent_id]
        value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
        # print(agent.cash, agent.position, agent.get_pos_value(), value)
        values.append(value)
    surpluses.append(sum(values)/len(values))
print(sum(surpluses)/len(surpluses)*25)