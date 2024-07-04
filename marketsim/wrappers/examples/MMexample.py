from marketsim.wrappers.MM_wrapper import MMEnv

def run():

    normalizers = {"fundamental": 1e5, "invt": 10, "cash": 5e5}
    beta_params = {'a_buy': 1, 'b_buy': 1, 'a_sell': 1, 'b_sell': 1}

    env = MMEnv(num_background_agents=100,
                sim_time=1000,
                lam=0.3,
                lamMM=0.1,
                mean=1e5,
                r=0.05,
                shock_var=5e6,
                q_max=10,
                pv_var=5e6,
                shade=[250,500],
                omega=10,
                normalizers=normalizers,
                beta_params=beta_params)

    obs, info = env.reset()

    for i in range(100):
        print("Current Iter:", i)
        print("Internal steps:", env.time)
        action = env.action_space.sample()  # this is where you would insert your policy
        # print(action)
        # print(env.step(action))

        observation, reward, terminated, truncated, info = env.step(action)
        print("Observations:", observation)
        print("Reward:", reward)
        print("terminated:", terminated)
        # print(env.markets[0].order_book.observe())
        print("---------------------------")


        if terminated or truncated:
            print("=========================")
            observation, info = env.reset()

    env.end_sim_summarize()

if __name__ == "__main__":
    run()