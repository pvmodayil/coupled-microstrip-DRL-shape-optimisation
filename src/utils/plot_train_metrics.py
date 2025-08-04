#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 04-08-2025
# Topic         : Training Metrics for RL agent
# Description   : Includes functions to extract and plot metrics from the tensorboard logging.
#################################################

#####################################################################################
#                                     Imports
#####################################################################################
# import pandas as pd
from datetime import datetime
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt

#####################################################################################
#                                     Functions
#####################################################################################
def plot_rewards(image_dir: str, log_file_path: str) -> None:
    # Create an EventAccumulator
    event_acc: EventAccumulator = EventAccumulator(log_file_path)
    event_acc.Reload()
    
    reward_data: list = event_acc.Scalars("rollout/ep_rew_mean")
    step_time: list = [event.wall_time for event in reward_data]
    steps: list[int] = [event.step for event in reward_data]
    rewards: list[float] = [event.value for event in reward_data]
    
    # Convert Unix timestamps to datetime objects
    timestamps_dt: list[datetime] = [datetime.fromtimestamp(unixtimestamps) for unixtimestamps in step_time]
    # Calculate relative hours from the first timestamp
    start_time: datetime = timestamps_dt[0]
    step_time_minutes: list[float] = [(ts - start_time).total_seconds() / 60 for ts in timestamps_dt]
    
    # Plot the reward with twin x-axis
    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    # Plot against steps_rew on ax1
    ax1.plot(steps, rewards, color='orange',label='_noLabel',linewidth=3)

    # Plot against relative_hours on ax2 (invisible for demonstration)
    ax2.plot(step_time_minutes, rewards, alpha=0)

    # Customize plot
    ax1.set_xlabel('Episode#', fontsize=25)
    ax1.set_ylabel("Rewards", fontsize=25)
    ax2.set_xlabel('Training Time [min]', fontsize=25)

    # Set ticks and labels
    ax1.tick_params(axis='x', labelsize=25, pad=10)
    ax1.tick_params(axis='y', labelsize=25, pad=10)
    ax2.tick_params(axis='x', labelsize=25, pad=10)

    # Display grid
    ax1.grid(True)
    
    # Show plot
    image_path: str = os.path.join(image_dir,'rewards.png')
    plt.savefig(image_path)
    plt.close()
    
def plot_entropy(image_dir: str, log_file_path: str) -> None:
    # Create an EventAccumulator
    event_acc: EventAccumulator = EventAccumulator(log_file_path)
    event_acc.Reload()
    
    entropy_data: list = event_acc.Scalars("train/ent_coef")
    steps: list[int] = [event.step for event in entropy_data]
    entropy_coeffs: list[float] = [event.value for event in entropy_data]
    
    # Plot the entropy with twin x-axis
    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(111)

    # Plot against steps_rew on ax1
    ax1.plot(steps, entropy_coeffs, color='orange',label='_noLabel',linewidth=3)

    # Customize plot
    ax1.set_xlabel('Episode#', fontsize=25)
    ax1.set_ylabel("Entropy Coefficient", fontsize=25)

    # Set ticks and labels
    ax1.tick_params(axis='x', labelsize=25, pad=10)
    ax1.tick_params(axis='y', labelsize=25, pad=10)

    # Display grid
    ax1.grid(True)
    
    # Show plot
    image_path: str = os.path.join(image_dir,'entropy.png')
    plt.savefig(image_path)
    plt.close()

def plot_loss(image_dir: str, log_file_path: str) -> None:
    # Create an EventAccumulator
    event_acc: EventAccumulator = EventAccumulator(log_file_path)
    event_acc.Reload()
    
    critic_loss_data: list = event_acc.Scalars("train/critic_loss")
    steps_critic_loss: list[int] = [event.step for event in critic_loss_data]
    critic_loss: list[float] = [event.value for event in critic_loss_data]
    
    actor_loss_data: list = event_acc.Scalars("train/actor_loss")
    steps_actor_loss: list[int] = [event.step for event in actor_loss_data]
    actor_loss: list[float] = [event.value for event in actor_loss_data]
    
    # Create a figure and axis objects
    fig, axs = plt.subplots(1, 2, figsize=(18, 12))

    # Flatten the axs array to easily access each subplot
    axs = axs.flatten()

    axs[0].plot(steps_critic_loss, critic_loss,color='orange',label='_Critic Loss',linewidth=3)
    axs[0].set_xlabel('Episode#',fontsize=25)
    axs[0].set_ylabel('Critic Loss',fontsize=25)
    axs[0].tick_params(axis='x', labelsize=25, pad=10)
    axs[0].tick_params(axis='y', labelsize=25, pad=10)

    axs[1].plot(steps_actor_loss, actor_loss,color='orange',label='_Actor Loss',linewidth=3)
    axs[1].set_xlabel('Episode#',fontsize=25)
    axs[1].set_ylabel('Actor Loss',fontsize=25)
    axs[1].tick_params(axis='x', labelsize=25, pad=10)
    axs[1].tick_params(axis='y', labelsize=25, pad=10)

    plt.tight_layout()
    
    
    # Show plot
    image_path: str = os.path.join(image_dir,'loss.png')
    plt.savefig(image_path)
    plt.close()
    
def main() -> None:
    pass

if __name__ == "__main__":
    main()