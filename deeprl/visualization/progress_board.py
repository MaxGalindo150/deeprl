import matplotlib.pyplot as plt
import torch
import matplotlib

class ProgressBoard:
    """A class that plots durations or cumulative rewards over episodes and saves the final plot."""

    def __init__(self, xlabel='Episode', ylabel='Duration', save_path=None):
        """
        Initialize the ProgressBoard.

        :param xlabel: Label for the x-axis (default is 'Episode').
        :param ylabel: Label for the y-axis (default is 'Duration').
        :param save_path: Path to save the final plot as an image file.
        """
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.save_path = save_path
        self.episode_durations = []

    def record(self, duration):
        """Store the duration data without drawing it."""
        self.episode_durations.append(duration)

    def draw(self):
        """Generate the plot using stored data and prepare for saving."""
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        
        plt.figure(1)
        plt.clf()
        plt.title('Training...')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

        # Plot cumulative reward with low opacity
        plt.plot(durations_t.numpy(), label='Cumulative Reward', color='blue', alpha=0.3)

        # Moving average for the last 100 episodes
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), label='100-Episode Moving Average', color='orange', linewidth=2)

        plt.legend()

    def save(self):
        """Save the final plot to a file if `save_path` is specified."""
        if self.save_path:
            # Draw the final plot before saving
            self.draw()
            plt.savefig(self.save_path)
            print(f"Plot saved to {self.save_path}")
