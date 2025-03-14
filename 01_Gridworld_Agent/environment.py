import numpy as np
import pygame

class GridWorld:
    def __init__(self, size=5, obstacles=[], goal=None):
        self.size =  size  # size of the grid (5x5 by default)
        self.grid = np.zeros((size, size))  # create grid with zeros

        # set the start and end points
        self.agent_pos = (0, 0)

        # set the goal point
        if goal:
            self.goal_pos = goal
        else:
            self.goal_pos = (size-1, size-1)

        # set the obstacles
        self.obstacles = obstacles

        # set the actions
        self.actions = [
            (0, 1), # right
            (0, -1), # left
            (1, 0), # down
            (-1, 0), # up
        ]

    def obstacle(self, obstacles):
        self.obstacles = obstacles

        for obs in self.obstacles:
            self.grid[obs] = -1

    def reset(self):
        # Clear the grid
        self.grid = np.zeros((self.size, self.size))

        # set the start and end points
        self.agent_pos = (0, 0)
        self.goal_pos = (self.size-1, self.size-1)

        # set the obstacles
        self.obstacle(self.obstacles)

        return self.agent_pos

    def step(self, action):
        # get the current position of the agent
        x, y = self.agent_pos

        # get the next position of the agent
        dx, dy = self.actions[action]
        next_x, next_y = x + dx, y + dy
        
        # check if the next position is valid
        if 0 <= next_x < self.size and 0 <= next_y < self.size and (next_x, next_y) not in self.obstacles:
            # update the position of the agent
            self.agent_pos = (next_x, next_y)
        
        # check if the agent has reached the goal
        done = self.agent_pos == self.goal_pos
        
        # get the reward
        if done:
            reward = 1.0
        elif self.agent_pos == (x, y):  # agent didn't move (hit wall or obstacle)
            reward = -1.0
        else:
            reward = -0.1  # step penalty
        
        return self.agent_pos, reward, done


    def render(self, cell_size=80):
        # Initialize pygame if not already done
        if not hasattr(self, 'pygame_initialized'):
            pygame.init()
            self.pygame_initialized = True
            self.screen = pygame.display.set_mode((self.size * cell_size, self.size * cell_size))
            pygame.display.set_caption("GridWorld")
            self.clock = pygame.time.Clock()
        
        # Fill background
        self.screen.fill((255, 255, 255))
        
        # Draw grid lines
        for i in range(self.size + 1):
            pygame.draw.line(self.screen, (200, 200, 200), (0, i * cell_size), (self.size * cell_size, i * cell_size), 2)
            pygame.draw.line(self.screen, (200, 200, 200), (i * cell_size, 0), (i * cell_size, self.size * cell_size), 2)
        
        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (100, 100, 100), 
                            (obs[1] * cell_size, obs[0] * cell_size, cell_size, cell_size))
        
        # Draw goal
        pygame.draw.rect(self.screen, (0, 255, 0), 
                        (self.goal_pos[1] * cell_size, self.goal_pos[0] * cell_size, cell_size, cell_size))
        
        # Draw agent
        agent_center = (self.agent_pos[1] * cell_size + cell_size // 2, 
                        self.agent_pos[0] * cell_size + cell_size // 2)
        pygame.draw.circle(self.screen, (255, 0, 0), agent_center, cell_size // 3)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(5)  # Limit to 5 frames per second
        
        # Process events - return True if quit was requested
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
        return False

def test_environment(env, episodes=3, max_steps=20):
    print("Testing GridWorld environment...")
    pygame_quit_requested = False
    
    try:   
        for episode in range(episodes):
            if pygame_quit_requested:
                break
                
            state = env.reset()
            print(f"\nEpisode {episode+1}")
            print(f"Initial state: {state}")
            
            for step in range(max_steps):
                # Take a random action
                action = np.random.randint(0, len(env.actions))
                action_name = ["right", "left", "down", "up"][action]
                
                # Get next state, reward, and done flag
                next_state, reward, done = env.step(action)
                
                print(f"Step {step+1}: Action={action_name}, New state={next_state}, Reward={reward}, Done={done}")
                
                # Render the environment and check if quit was requested
                pygame_quit_requested = env.render()
                if pygame_quit_requested:
                    break
                
                # If episode is done, break
                if done:
                    print(f"Goal reached in {step+1} steps!")
                    break
                
                # Add a small delay to see the movement
                pygame.time.delay(500)
            
            if not done and not pygame_quit_requested:
                print(f"Goal not reached after {max_steps} steps.")
                
        if not pygame_quit_requested:
            print("\nEnvironment test complete!")
        
    finally:
        # Make sure pygame quits properly even if there was an error
        if pygame.get_init():
            pygame.quit()

if __name__ == "__main__":
    env = GridWorld(size=5, obstacles=[(1, 1), (2, 2), (3, 1), (3, 2)])
    test_environment(env)