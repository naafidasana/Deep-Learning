import numpy as np
import pygame
import rich

class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.995, epsilon_decay=0.99, min_epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((env.size, env.size, len(env.actions)))

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(len(self.env.actions))
        else:
            return np.argmax(self.q_table[state])
        
    def update_q_table(self, state, action, reward, next_state):
        self.q_table[state][action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action])   

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            print(f"Episode {episode+1}/{episodes}, Total reward: {total_reward}, Epsilon: {self.epsilon}")

    def test_agent(self, episodes=5, max_steps=20):
        print("Testing GridWorld environment...")
        pygame_quit_requested = False
        
        for episode in range(episodes):
            if pygame_quit_requested:
                break
                
            state = self.env.reset()
            print(f"\nEpisode {episode+1}")
            print(f"Initial state: {state}")
            
            for step in range(max_steps):
                # Take best action
                action = np.argmax(self.q_table[state])
                action_name = ["right", "left", "down", "up"][action]
                
                # Get next state, reward, and done flag
                next_state, reward, done = self.env.step(action)
                
                print(f"Step {step+1}: Action={action_name}, New state={next_state}, Reward={reward}, Done={done}")
                
                # Render the environment
                pygame_quit_requested = self.env.render()
                if pygame_quit_requested:
                    break
                
                # If episode is done, break
                if done:
                    print(f"Goal reached in {step+1} steps!")
                    break
                
                # Add a small delay to see the movement
                pygame.time.delay(500)
                
                # Update state
                state = next_state
            
            if not done and not pygame_quit_requested:
                print(f"Goal not reached after {max_steps} steps.")
        
        if not pygame_quit_requested:
            print("\nEnvironment test complete!")

    def visualize_q_table_as_grid(self):
        """Visualize the Q-table as a grid with all action values for each state."""
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        table = Table(title="Detailed Q-table Grid")
        
        # Add columns
        action_names = ["RIGHT", "LEFT", "DOWN", "UP"]
        table.add_column("State", style="cyan")
        for action in action_names:
            table.add_column(action, style="green")
        
        # Add rows for each state
        for i in range(self.env.size):
            for j in range(self.env.size):
                state = (i, j)
                if state == self.env.goal_pos:
                    state_name = f"({i},{j})-GOAL"
                    style = "bold green"
                elif state in self.env.obstacles:
                    state_name = f"({i},{j})-OBST"
                    style = "dim"
                else:
                    state_name = f"({i},{j})"
                    style = "white"
                    
                q_values = [f"{q:.2f}" for q in self.q_table[state]]
                table.add_row(state_name, *q_values, style=style)
        
        console.print(table)

    def visualize_best_actions_grid(self):
        """Visualize the best action and its Q-value for each state in a grid."""
        from rich.console import Console
        from rich.table import Table
        
        action_symbols = ["→", "←", "↓", "↑"]
        console = Console()
        table = Table(title="Best Actions Grid", show_header=False, box=rich.box.SQUARE)
        
        # Add rows for each state
        for i in range(self.env.size):
            row_cells = []
            for j in range(self.env.size):
                state = (i, j)
                if state == self.env.goal_pos:
                    cell = "[bold green]G[/bold green]"
                elif state in self.env.obstacles:
                    cell = "[dim]#[/dim]"
                else:
                    best_action = np.argmax(self.q_table[state])
                    best_q = self.q_table[state][best_action]
                    
                    # Color-code based on Q-value
                    if best_q > 0.5:
                        color = "green"
                    elif best_q > 0:
                        color = "yellow"
                    else:
                        color = "red"
                        
                    cell = f"[{color}]{action_symbols[best_action]}[/{color}]"
                row_cells.append(cell)
            table.add_row(*row_cells)
        
        console.print(table)

    def visualize_policy_pygame(self, cell_size=80, wait_time=0):
        """Visualize the learned policy using pygame."""
        action_vectors = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        action_colors = [(0, 0, 255), (255, 0, 255), (255, 165, 0), (0, 255, 255)]
        
        # Initialize pygame if not already done
        if not hasattr(self.env, 'pygame_initialized'):
            pygame.init()
            self.env.pygame_initialized = True
            self.env.screen = pygame.display.set_mode((self.env.size * cell_size, self.env.size * cell_size))
            pygame.display.set_caption("GridWorld Policy")
            self.env.clock = pygame.time.Clock()
        
        # Fill background
        self.env.screen.fill((255, 255, 255))
        
        # Draw grid lines
        for i in range(self.env.size + 1):
            pygame.draw.line(self.env.screen, (200, 200, 200), 
                            (0, i * cell_size), (self.env.size * cell_size, i * cell_size), 2)
            pygame.draw.line(self.env.screen, (200, 200, 200), 
                            (i * cell_size, 0), (i * cell_size, self.env.size * cell_size), 2)
        
        # Draw obstacles
        for obs in self.env.obstacles:
            pygame.draw.rect(self.env.screen, (100, 100, 100), 
                            (obs[1] * cell_size, obs[0] * cell_size, cell_size, cell_size))
        
        # Draw goal
        pygame.draw.rect(self.env.screen, (0, 255, 0), 
                        (self.env.goal_pos[1] * cell_size, self.env.goal_pos[0] * cell_size, cell_size, cell_size))
        
        # Draw policy arrows
        font = pygame.font.SysFont(None, cell_size // 3)
        
        for i in range(self.env.size):
            for j in range(self.env.size):
                state = (i, j)
                if state == self.env.goal_pos or state in self.env.obstacles:
                    continue
                    
                # Get best action and its Q-value
                best_action = np.argmax(self.q_table[state])
                best_q = self.q_table[state][best_action]
                
                # Draw arrow for best action
                center_x = j * cell_size + cell_size // 2
                center_y = i * cell_size + cell_size // 2
                
                # Calculate arrow endpoint
                dx, dy = action_vectors[best_action]
                end_x = center_x + dx * (cell_size // 3)
                end_y = center_y + dy * (cell_size // 3)
                
                # Draw arrow
                pygame.draw.line(self.env.screen, action_colors[best_action], 
                            (center_x, center_y), (end_x, end_y), 4)
                
                # Draw arrowhead
                if best_action == 0:  # right
                    pygame.draw.polygon(self.env.screen, action_colors[best_action], 
                                    [(end_x, end_y), (end_x-8, end_y-8), (end_x-8, end_y+8)])
                elif best_action == 1:  # left
                    pygame.draw.polygon(self.env.screen, action_colors[best_action], 
                                    [(end_x, end_y), (end_x+8, end_y-8), (end_x+8, end_y+8)])
                elif best_action == 2:  # down
                    pygame.draw.polygon(self.env.screen, action_colors[best_action], 
                                    [(end_x, end_y), (end_x-8, end_y-8), (end_x+8, end_y-8)])
                elif best_action == 3:  # up
                    pygame.draw.polygon(self.env.screen, action_colors[best_action], 
                                    [(end_x, end_y), (end_x-8, end_y+8), (end_x+8, end_y+8)])
                
                # Draw Q-value
                q_text = font.render(f"{best_q:.2f}", True, (0, 0, 0))
                self.env.screen.blit(q_text, (center_x - q_text.get_width()//2, 
                                            center_y + cell_size//4))
        
        # Update display
        pygame.display.flip()
        
        # If wait_time provided, wait then return
        if wait_time > 0:
            start_time = pygame.time.get_ticks()
            while pygame.time.get_ticks() - start_time < wait_time:
                # Process events while waiting
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return True
                pygame.time.delay(100)  # Short delay to prevent CPU hogging
        
        # Process events - return True if quit was requested
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
        return False