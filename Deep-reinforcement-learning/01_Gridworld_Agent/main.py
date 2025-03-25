from environment import GridWorld
from q_learning import QLearning
import pygame

if __name__ == "__main__":
    try:
        # Create and train agent
        env = GridWorld(size=5, obstacles=[(1, 1), (2, 2), (3, 1), (3, 2)])
        agent = QLearning(env)
        agent.train(episodes=500)
        
        # Show visualizations
        agent.visualize_q_table_as_grid()
        agent.visualize_best_actions_grid()
        
        # Test and visualize policy
        agent.test_agent(episodes=5)
        agent.visualize_policy_pygame(wait_time=5000)  # Show policy for 5 seconds
        
    finally:
        # Make sure pygame quits properly even if there was an error
        if pygame.get_init():
            pygame.quit()