import pygame
import numpy as np
import random
from scipy.ndimage import convolve

# Define constants
GRID_SIZE = 8
CELL_SIZE = 64
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (192, 192, 192)  # For Brian's Brain

# Define the maximum number of channels at the top of your script
MAX_CHANNELS = 8  # or any number that suits your requirements
FREQ = 8000  # Same as audio CD quality 8000, 16000, 22050, 24000, 32000, 44100, 48000
BUFFER = 256  # 4096 is a better buffer size but may result in sound lag

# Frame rates for each rule set
FRAMERATE_CONWAY = 4
FRAMERATE_WOLFRAM = 4
FRAMERATE_BRIANS_BRAIN = 4
FRAMERATE_LENIA = 4 

# Define a maximum volume for each channel
max_channel_volume = 0.1  # 50% of the maximum volume
volume_level_index = 5  # Corresponds to a volume level of 0.5
volume_steps = [i / 10 for i in range(0, 11)]  # [0, 0.1, 0.2, ..., 1.0]

# Grid to hold the state of the cells
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
# Rule set: 0 for Conway's Game of Life, 1 for Wolfram ECA, 2 for Brian's Brain
rule_set = 0


# Menu class definition
class Menu:
    def __init__(self, screen, font, framerate=30):  # default framerate set to 10
        self.screen = screen
        self.font = font
        self.framerate = framerate 
        self.running = True
        self.grid = None  # This will hold the cellular automaton grid
        self.init_grid()  # Initialize the grid with a random state
        self.constants = {
            'GRID_SIZE': GRID_SIZE,
            'CELL_SIZE': CELL_SIZE,
            'FRAMERATE_CONWAY': FRAMERATE_CONWAY,
            'FRAMERATE_WOLFRAM': FRAMERATE_WOLFRAM,
            'FRAMERATE_BRIANS_BRAIN': FRAMERATE_BRIANS_BRAIN,
            'FRAMERATE_LENIA': FRAMERATE_LENIA,
            'MAX_CHANNELS': MAX_CHANNELS,
            'FREQ': FREQ
        }
        self.selected_constant = list(self.constants.keys())[0]
        self.index = 0

    def init_grid(self):
        # Initialize a grid with a random seed
        self.grid = np.random.randint(2, size=(GRID_SIZE, GRID_SIZE), dtype=int)

    def run_menu(self):
        clock = pygame.time.Clock()  # Create a clock object
        while self.running:
            clock.tick(self.framerate) 
            self.screen.fill((0, 0, 0))
            self.update_conway()  # Update the grid using the Conway's Game of Life rules
            self.draw_grid()  # Draw the grid as the background
            self.draw_menu()  # Draw the menu on top of the grid
            
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.start_synthesizer()
                    elif event.key in [pygame.K_UP, pygame.K_DOWN]:
                        self.change_selection(event.key)
                    elif event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                        self.update_constant(event.key)
                    elif event.key == pygame.K_ESCAPE:  # Handle ESC key in the menu
                        self.running = False
                        pygame.quit()
                        quit()

    def update_conway(self):
        new_grid = np.zeros_like(self.grid)
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                live_neighbors = self.get_live_neighbors(x, y)
                if self.grid[y, x] == 1 and live_neighbors in [2, 3]:
                    new_grid[y, x] = 1
                elif self.grid[y, x] == 0 and live_neighbors == 3:
                    new_grid[y, x] = 1
        self.grid = new_grid

    def get_live_neighbors(self, x, y):
        # This method should be the same as in the main script.
        # It calculates the number of live neighbors for a cell in Conway's Game of Life.
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = (x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE
                count += self.grid[ny, nx] == 1
        return count

    def draw_grid(self):
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                color = WHITE if self.grid[y, x] == 1 else BLACK
                # Create a new surface with per-pixel alpha
                cell_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                # Set the alpha value for the cell color (0-255, 0 is fully transparent, 255 is opaque)
                alpha_value = 128  # Example: Half transparent
                # Add the alpha value to the color tuple
                cell_color = color + (alpha_value,)
                # Fill the cell surface with the translucent color
                cell_surface.fill(cell_color)
                # Get the position for where to blit the cell surface on the main screen
                pos = (x * CELL_SIZE, y * CELL_SIZE)
                # Blit the cell surface onto the main screen surface
                self.screen.blit(cell_surface, pos)

    def draw_menu(self):
        menu_text = 'synthOMATA: Press Enter to start'
        text_surface = self.font.render(menu_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (50, 50))

        for i, (key, value) in enumerate(self.constants.items()):
            text_color = (255, 255, 0) if self.index == i else (255, 255, 255)
            constant_surface = self.font.render(f'{key}: {value}', True, text_color)
            self.screen.blit(constant_surface, (50, 100 + i * 30))

    def change_selection(self, key):
        if key == pygame.K_UP:
            self.index = (self.index - 1) % len(self.constants)
        elif key == pygame.K_DOWN:
            self.index = (self.index + 1) % len(self.constants)
        self.selected_constant = list(self.constants.keys())[self.index]

    def update_constant(self, key):
        # Cycle through predefined frequency values
        predefined_freqs = [8000, 16000, 22050, 24000, 32000, 44100, 48000]
        value = self.constants[self.selected_constant]

        if self.selected_constant == 'FREQ':
            if key == pygame.K_RIGHT:
                # Get the next frequency in the list, or the first one if currently at the last one
                value = predefined_freqs[(predefined_freqs.index(value) + 1) % len(predefined_freqs)]
            elif key == pygame.K_LEFT:
                # Get the previous frequency in the list, or the last one if currently at the first one
                value = predefined_freqs[(predefined_freqs.index(value) - 1) % len(predefined_freqs)]
        else:
            # For other constants, increase or decrease the value
            if key == pygame.K_RIGHT:
                value += 1
            elif key == pygame.K_LEFT:
                value = max(0, value - 1)
        
        self.constants[self.selected_constant] = value

    def start_synthesizer(self):
        self.running = False
        # Start the synthesizer with the updated constants
        run_synthesizer(self.constants)


def run_synthesizer(settings):
    # Unpack settings
    global WINDOW_WIDTH, WINDOW_HEIGHT
    global GRID_SIZE, CELL_SIZE, FRAMERATE_CONWAY, FRAMERATE_WOLFRAM, FRAMERATE_BRIANS_BRAIN, MAX_CHANNELS, FREQ
    GRID_SIZE = settings['GRID_SIZE']
    CELL_SIZE = settings['CELL_SIZE']
    FRAMERATE_CONWAY = settings['FRAMERATE_CONWAY']
    FRAMERATE_WOLFRAM = settings['FRAMERATE_WOLFRAM']
    FRAMERATE_BRIANS_BRAIN = settings['FRAMERATE_BRIANS_BRAIN']
    FRAMERATE_LENIA = settings['FRAMERATE_LENIA']
    MAX_CHANNELS = settings['MAX_CHANNELS']
    FREQ = settings['FREQ']

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("synthOMATA")

    # Grid to hold the state of the cells
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    # Update the window size
    WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
    WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE

    # Reset and resize the grid and window
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    pygame.mixer.init(frequency=FREQ, size=-16, channels=64, buffer=BUFFER)
    pygame.mixer.set_num_channels(MAX_CHANNELS)


cell_to_channel = {}  # Dictionary to map cells to their sound channels

def manage_cell_sounds(grid, frequencies, cell_to_channel, grid_width, grid_height, automaton_type, old_grid=None):
    for y in range(grid_height):
        for x in range(grid_width):
            cell_index = (y, x)

            if automaton_type in ['conway', 'wolfram', 'brians_brain', 'lenia']:
                cell_state = grid[y][x]
                # If the cell is on and doesn't have a sound playing, start a sound
                if cell_state == 1 and cell_index not in cell_to_channel:
                    frequency = calculate_frequency_for_cell(x, y, grid_width, grid_height, frequencies)
                    channel = play_frequency(frequency)
                    cell_to_channel[cell_index] = channel
                # If the cell is off and has a sound playing, stop the sound
                elif cell_state == 0 and cell_index in cell_to_channel:
                    channel = cell_to_channel.pop(cell_index)
                    channel.stop()

    # Clean up finished channels
    for cell_index, channel in list(cell_to_channel.items()):
        if channel is None or not channel.get_busy():
            del cell_to_channel[cell_index]


def play_frequency(frequency):
    sample_rate = 22050  # Sample rate in Hz
    period_samples = int(sample_rate / frequency)
    # Create a buffer for one period of the wave
    buf_mono = np.sin(2 * np.pi * np.arange(period_samples) * frequency / sample_rate).astype(np.float32)
    # Normalize the buffer
    max_val = np.max(np.abs(buf_mono))
    if max_val > 0:
        buf_mono /= max_val
    # Convert to 16-bit PCM
    buf_mono = (buf_mono * 32767).astype(np.int16)
    # Duplicate the mono buffer into a stereo buffer (2D array)
    buf_stereo = np.repeat(buf_mono[:, np.newaxis], 2, axis=1)
    # Create a sound object from the stereo buffer
    sound = pygame.sndarray.make_sound(buf_stereo)
    if sound is None:
        print(f"Failed to create sound for frequency: {frequency}")
        return None
    channel = sound.play(-1)  # Play the sound indefinitely until stopped
    if channel:
        # Set the volume of this channel to not exceed the maximum
        channel.set_volume(max_channel_volume)
    else:
        #print(f"Failed to play sound for frequency: {frequency}")
        pass

    return channel


def calculate_lydian_frequencies(base_freq, num_octaves):
    # The ratio between the frequencies of two consecutive notes in a scale
    # is the twelfth root of 2
    ratio = 2 ** (1/12)
    
    # C Lydian scale intervals (W-W-W-H-W-W-H)
    intervals = [2, 2, 2, 1, 2, 2, 1]
    
    # Starting with the base frequency, calculate the frequency for each note
    # in the Lydian mode across the specified number of octaves
    frequencies = [base_freq]
    current_freq = base_freq
    for octave in range(num_octaves):
        for step in intervals:
            current_freq *= ratio**step
            frequencies.append(current_freq)
    return frequencies

# Let's assume you want to start from C2 (65.41 Hz)
# and cover 7 octaves to reach beyond C8 (4186 Hz)
base_freq = 65.41  # Frequency of C2
num_octaves = 7
lydian_frequencies = calculate_lydian_frequencies(base_freq, num_octaves)


def calculate_frequency_for_cell(x, y, grid_width, grid_height, frequencies):
    # Inverting the y-coordinate so the bottom left is 0,0 in Pygame's coordinate system
    index = (grid_height - y - 1) * grid_width + x
    # Ensure the index does not exceed the length of the frequencies list
    freq_index = index % len(frequencies)
    return frequencies[freq_index]


def update_volume(volume, cell_to_channel):
    for channel in cell_to_channel.values():
        if channel:
            channel.set_volume(volume)




def play_grid_tones(grid, frequencies):
    grid_height = len(grid)
    grid_width = len(grid[0])

    for y in range(grid_height):
        for x in range(grid_width):
            if grid[y][x] == 1:  # If cell is alive
                frequency = calculate_frequency_for_cell(x, y, grid_width, grid_height, frequencies)
                play_frequency(frequency, duration=1)


starting_octave = 2  # Starting from C2
base_freq_C4 = 261.63  # Frequency of C4
octaves_below_C4 = 4 - starting_octave

# Calculate the starting frequency based on how many octaves below C4 we want to start
base_freq = base_freq_C4 / (2 ** octaves_below_C4)

# Now call the function to calculate the Lydian frequencies with the new base frequency
lydian_frequencies = calculate_lydian_frequencies(base_freq, num_octaves)


# Function to draw the grid
def draw_grid(grid_size, cell_size, rule_set, screen):
    for y in range(grid_size):
        for x in range(grid_size):
            if rule_set == 2 and grid[y % GRID_SIZE, x % GRID_SIZE] == 2:  # Brian's Brain "firing" state
                color = GRAY
            else:
                color = WHITE if grid[y % GRID_SIZE, x % GRID_SIZE] == 1 else BLACK
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)

# Update functions for each automaton
def update_conway(grid):
    new_grid = np.zeros_like(grid)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            live_neighbors = get_live_neighbors(x, y)
            if grid[y, x] == 1 and live_neighbors in [2, 3]:
                new_grid[y, x] = 1
            elif grid[y, x] == 0 and live_neighbors == 3:
                new_grid[y, x] = 1
    return new_grid

def update_lenia(grid, growth_rate=0.1, decay_rate=0.01, cannibalize_rate=0.05, threshold=0.5):
    kernel = np.array([[0, 1, 2], 
                       [1, 4, 1], 
                       [2, 1, 1]])
    kernel = kernel / kernel.sum()

    # Apply convolution to the grid
    convolved = convolve(grid, kernel, mode='wrap')

    # Non-linear transformation
    transformed = 1 / (1 + np.exp(-convolved + threshold))

    # Growth and decay
    new_grid = grid + growth_rate * transformed - decay_rate * grid

    # Cannibalization
    cannibalized = convolve(grid > threshold, kernel, mode='wrap')
    new_grid -= cannibalize_rate * cannibalized

    # Clipping the grid values
    new_grid = np.clip(new_grid, 0, 1)


    return new_grid


def update_wolfram(grid, rule_number):
    new_grid = np.zeros_like(grid)
    new_grid[0] = grid[0]  # Copy the first row as is
    for y in range(1, grid.shape[0]):
        new_grid[y] = apply_rule(grid[y-1], rule_number)
    return new_grid

def update_brians_brain(grid):
    new_grid = np.zeros_like(grid)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            live_neighbors = get_live_neighbors(x, y)
            if grid[y, x] == 0 and live_neighbors == 2:  # Dead cell with two live neighbors comes to life
                new_grid[y, x] = 1
            elif grid[y, x] == 1:  # Live cell dies (fires)
                new_grid[y, x] = 2
            elif grid[y, x] == 2:  # Dying cell becomes dead
                new_grid[y, x] = 0
    return new_grid

# Function to create an explosion effect at a given position
def create_explosion(pos, velocity, grid_size):
    radius = int(velocity / 5) + 1  # The radius of the explosion
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = (pos[0] // CELL_SIZE + dx) % grid_size, (pos[1] // CELL_SIZE + dy) % grid_size
            if 0 <= x < grid_size and 0 <= y < grid_size:
                grid[y, x] ^= 1  # Flip the state of the cell



def create_explosion_brians_brain(pos, velocity, grid_size):
    radius = int(np.sqrt(velocity)) + 1  # Radius based on the square root of velocity
    x0, y0 = (pos[0] // CELL_SIZE), (pos[1] // CELL_SIZE)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx**2 + dy**2 <= radius**2:  # Check for circular area
                x, y = (x0 + dx) % grid_size, (y0 + dy) % grid_size
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    grid[y, x] = 1 if grid[y, x] == 0 else 0


# Function to get the number of live neighbors for a cell (for Conway's Game of Life and Brian's Brain)
def get_live_neighbors(x, y):
    count = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = (x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE
            count += grid[ny, nx] == 1
    return count

def apply_rule(row, rule_number):
    rule_string = "{:08b}".format(rule_number)
    new_row = np.zeros_like(row, dtype=int)  # Ensure new_row is of integer type
    row = row.astype(int)  # Convert row to integer type

    for i in range(1, len(row) - 1):
        # Convert the three cells into a number between 0 and 7
        neighborhood = (row[i-1] << 2) | (row[i] << 1) | row[i+1]
        # Apply the rule
        new_row[i] = int(rule_string[7 - neighborhood])

    return new_row

# Function to generate frame data from the grid
def generate_frame_data(grid, cell_size, grid_size):
    frame_data = np.zeros((grid_size * cell_size, grid_size * cell_size, 3), dtype=np.uint8)
    for y in range(grid_size):
        for x in range(grid_size):
            color = WHITE if grid[y, x] == 1 else BLACK
            frame_data[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size] = color
    return frame_data

# Initialize previous mouse position to None
prev_mouse_position = None

# Create a clock object
clock = pygame.time.Clock()

# Main application entry point
if __name__ == "__main__":
    pygame.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("synthoMATA")
    font = pygame.font.SysFont("Courier New", 16)
    menu = Menu(screen, font)
    menu.run_menu()

    # Update the grid and window size after the menu
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
# Main loop
running = True
while running:

    # Save old grid state for comparison
    old_grid = np.copy(grid)

    # Update the grid and manage sounds
    if rule_set == 0:  # Conway's Game of Life
        clock.tick(FRAMERATE_CONWAY)
        grid = update_conway(grid)
        manage_cell_sounds(grid, lydian_frequencies, cell_to_channel, GRID_SIZE, GRID_SIZE, 'conway')
    elif rule_set == 1:  # Wolfram's Elementary Cellular Automata
        clock.tick(FRAMERATE_WOLFRAM)
        grid = update_wolfram(grid, rule_number)
        manage_cell_sounds(grid, lydian_frequencies, cell_to_channel, GRID_SIZE, GRID_SIZE, 'wolfram')
    elif rule_set == 2:  # Brian's Brain
        clock.tick(FRAMERATE_BRIANS_BRAIN)
        grid = update_brians_brain(grid)
        manage_cell_sounds(grid, lydian_frequencies, cell_to_channel, GRID_SIZE, GRID_SIZE, 'brians_brain')
    elif rule_set == 3:  # Lenia
        
        # refresh the grid
             
        clock.tick(FRAMERATE_LENIA)
        grid = update_lenia(grid)
        for channel in cell_to_channel.values():
            
            old_grid = np.copy(grid)
            
            if channel:
                #modulate the channel volume
                channel.set_volume(0.1 * FRAMERATE_LENIA)
                
                channel.stop()
        manage_cell_sounds(grid, lydian_frequencies, cell_to_channel, GRID_SIZE, GRID_SIZE, 'lenia', old_grid)


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:  # Handle ESC key
                running = False
                return_to_menu = True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_click_pos = pygame.mouse.get_pos()
            if event.button == 1:  # Left click
                # Calculate velocity
                velocity = np.linalg.norm(np.array(mouse_click_pos) - np.array(prev_mouse_position if prev_mouse_position else mouse_click_pos))
                prev_mouse_position = mouse_click_pos
                if rule_set in [0, 2, 3]:  # For Conway and Brian's Brain, apply explosion effect
                    create_explosion_brians_brain(mouse_click_pos, velocity, GRID_SIZE)
                elif rule_set == 1:  # If it's Wolfram ECA, change the rule number
                    rule_number = random.randint(0, 255)
            elif event.button == 3:  # Right click, change rule type
                rule_set = (rule_set + 1) % 4
                #grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # Reset grid when switching rule sets
                if rule_set == 1:  # If switching to Wolfram ECA, choose a random rule number
                    rule_number = random.randint(0, 255)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_v:
                volume_level_index = (volume_level_index + 1) % len(volume_steps)
                update_volume(volume_steps[volume_level_index], cell_to_channel)

    



    # Draw the current state of the grid
    screen.fill(BLACK)
    draw_grid(GRID_SIZE, CELL_SIZE, rule_set, screen)
    pygame.display.flip()




# After the main loop
if return_to_menu:
    menu.run_menu()
else:
    pygame.quit()
