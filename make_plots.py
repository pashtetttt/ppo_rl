import matplotlib.pyplot as plt
import numpy as np
import re

def parse_ppo_logs(log_string):
    """
    Parse PPO training logs into separate lists.
    
    Args:
        log_string: Multi-line string containing PPO training output
        
    Returns:
        tuple: (ppo_steps, episodes, last_returns, policy_losses, value_losses)
    """
    ppo_steps = []
    episodes = []
    last_returns = []
    policy_losses = []
    value_losses = []
    value_losses_epochs = []
    
    # Split into lines and process each line
    lines = log_string.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Parse step lines (contain "step", "episodes", "return")
        if "step" in line and "return" in line:
            # Extract step number
            step_match = re.search(r'step\s+(\d+)', line)
            if step_match:
                ppo_steps.append(int(step_match.group(1)))
            
            # Extract episodes
            episodes_match = re.search(r'episodes\s+(\d+)', line)
            if episodes_match:
                episodes.append(int(episodes_match.group(1)))
            
            # Extract last return
            return_match = re.search(r'return=(-?\d+\.?\d*)', line)
            if return_match:
                last_returns.append(float(return_match.group(1)))
        
        # Parse update lines (contain "update", "policy_loss", "value_loss")
        elif "update" in line and "policy_loss" in line:
            # Extract policy loss
            policy_match = re.search(r'policy_loss=(-?\d+\.?\d*)', line)
            if policy_match:
                policy_losses.append(float(policy_match.group(1)))
            
            # Extract value loss
            value_match = re.search(r'value_loss=(-?\d+\.?\d*)', line)
            if value_match:
                value_losses.append(float(value_match.group(1)))
                value_losses_epochs.append(episodes[-1])
    
    return ppo_steps, episodes, last_returns, policy_losses, value_losses, value_losses_epochs


def plot_training_results(episodes, last_returns, policy_losses=None, value_losses=None, value_losses_epochs=None,
                         title="PPO Training Results", save_path=None):
    """
    Create plots for training returns and losses.
    
    Args:
        episodes: List of episode numbers
        last_returns: List of returns per episode
        policy_losses: List of policy losses (optional)
        value_losses: List of value losses (optional)
        title: Main title for the plot
        save_path: If provided, save figure to this path
    """
    # Create figure with subplots
    if policy_losses is not None or value_losses is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        ax2 = None
    
    # Plot returns on first subplot
    ax1.plot(episodes, last_returns, 'b-', linewidth=2, label=f'Episode Return, last = {last_returns[-1]}')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Return', fontsize=12)
    ax1.set_title(f'{title} - Returns over Episodes', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # # Add mean return line and annotation
    # if len(last_returns) > 0:
    #     mean_return = np.mean(last_returns[-min(50, len(last_returns)):])  # Mean of last 50 episodes
    #     ax1.axhline(y=mean_return, color='r', linestyle='--', alpha=0.7, 
    #                label=f'Mean Return (last 50): {mean_return:.2f}')
    #     ax1.legend(loc='best')

    # Add best return line and annotation
    if len(last_returns) > 0:
        best_return = 21  # Mean of last 50 episodes
        ax1.axhline(y=best_return, color='r', linestyle='--', alpha=0.7, 
                   label=f'Best Return (goal): {best_return:.2f}')
        ax1.legend(loc='best')
    
    # Plot losses on second subplot if provided
    if ax2 is not None:
        if policy_losses is not None:
            # Policy losses might be recorded at different frequency than episodes
            # Create x-axis indices for losses
            loss_indices = value_losses_epochs
            ax2.plot(loss_indices, policy_losses, 'g-', linewidth=2, label='Policy Loss')
        
        if value_losses is not None:
            loss_indices = value_losses_epochs
            ax2.plot(loss_indices, value_losses, 'orange', linewidth=2, label='Value Loss')
        
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title(f'{title} - Losses over Episodes', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    """
    Main function to parse logs and create plots.
    """
    # Your log string
    log_string = """
  PPO step 4199  episodes 5  last return=-21  (agent 0 - opp 21)
  PPO step 8751  episodes 10  last return=-21  (agent 0 - opp 21)
  PPO step 13272  episodes 15  last return=-21  (agent 0 - opp 21)
  PPO step 17545  episodes 20  last return=-21  (agent 0 - opp 21)
  PPO step 21766  episodes 25  last return=-20  (agent 1 - opp 21)
  PPO update  policy_loss=-0.0026  value_loss=0.0701
  PPO step 26479  episodes 30  last return=-20  (agent 1 - opp 21)
  PPO update  policy_loss=-0.0013  value_loss=0.0738
  PPO step 30935  episodes 35  last return=-21  (agent 0 - opp 21)
  PPO step 35546  episodes 40  last return=-20  (agent 1 - opp 21)
  PPO step 39994  episodes 45  last return=-21  (agent 0 - opp 21)
  PPO step 44257  episodes 50  last return=-21  (agent 0 - opp 21)
  PPO update  policy_loss=-0.0012  value_loss=0.0685
  PPO step 48551  episodes 55  last return=-21  (agent 0 - opp 21)
  PPO update  policy_loss=-0.0037  value_loss=0.0398
  PPO step 52770  episodes 60  last return=-21  (agent 0 - opp 21)
  PPO update  policy_loss=-0.0009  value_loss=0.0508
  PPO step 57197  episodes 65  last return=-19  (agent 2 - opp 21)
  PPO update  policy_loss=-0.0007  value_loss=0.0665
  PPO step 61501  episodes 70  last return=-21  (agent 0 - opp 21)
  PPO step 66415  episodes 75  last return=-21  (agent 0 - opp 21)
  PPO step 71028  episodes 80  last return=-21  (agent 0 - opp 21)
  PPO update  policy_loss=0.0000  value_loss=0.0371
  PPO step 75424  episodes 85  last return=-21  (agent 0 - opp 21)
  PPO update  policy_loss=-0.0053  value_loss=0.0519
  PPO step 80074  episodes 90  last return=-20  (agent 1 - opp 21)
  PPO step 84526  episodes 95  last return=-21  (agent 0 - opp 21)
  PPO step 89065  episodes 100  last return=-20  (agent 1 - opp 21)
  PPO step 93393  episodes 105  last return=-21  (agent 0 - opp 21)
  PPO step 97890  episodes 110  last return=-21  (agent 0 - opp 21)
  PPO update  policy_loss=-0.0009  value_loss=0.0378
  PPO step 102561  episodes 115  last return=-19  (agent 2 - opp 21)
  PPO step 106983  episodes 120  last return=-20  (agent 1 - opp 21)
  PPO step 111044  episodes 125  last return=-21  (agent 0 - opp 21)
  PPO step 115338  episodes 130  last return=-21  (agent 0 - opp 21)
  PPO step 119745  episodes 135  last return=-21  (agent 0 - opp 21)
  PPO step 124120  episodes 140  last return=-19  (agent 2 - opp 21)
  PPO update  policy_loss=-0.0025  value_loss=0.0587
  PPO step 129076  episodes 145  last return=-18  (agent 3 - opp 21)
  PPO step 134018  episodes 150  last return=-20  (agent 1 - opp 21)
  PPO step 139024  episodes 155  last return=-16  (agent 5 - opp 21)
  PPO update  policy_loss=-0.0050  value_loss=0.0908
  PPO step 143748  episodes 160  last return=-19  (agent 2 - opp 21)
  PPO step 148520  episodes 165  last return=-20  (agent 1 - opp 21)
  PPO step 153460  episodes 170  last return=-20  (agent 1 - opp 21)
  PPO update  policy_loss=-0.0010  value_loss=0.0571
  PPO step 158391  episodes 175  last return=-20  (agent 1 - opp 21)
  PPO step 163542  episodes 180  last return=-19  (agent 2 - opp 21)
  PPO update  policy_loss=-0.0013  value_loss=0.0558
  PPO step 168358  episodes 185  last return=-21  (agent 0 - opp 21)
  PPO step 173301  episodes 190  last return=-20  (agent 1 - opp 21)
  PPO update  policy_loss=-0.0002  value_loss=0.0464
  PPO step 178347  episodes 195  last return=-21  (agent 0 - opp 21)
  PPO step 183531  episodes 200  last return=-17  (agent 4 - opp 21)
  PPO update  policy_loss=-0.0021  value_loss=0.0631
  PPO step 188377  episodes 205  last return=-20  (agent 1 - opp 21)
  PPO update  policy_loss=-0.0032  value_loss=0.0495
  PPO step 193829  episodes 210  last return=-19  (agent 2 - opp 21)
  PPO update  policy_loss=-0.0019  value_loss=0.0507
  PPO step 199048  episodes 215  last return=-17  (agent 4 - opp 21)
  PPO step 204723  episodes 220  last return=-20  (agent 1 - opp 21)
  PPO update  policy_loss=-0.0025  value_loss=0.0474
  PPO step 210466  episodes 225  last return=-20  (agent 1 - opp 21)
  PPO update  policy_loss=-0.0017  value_loss=0.0572
  PPO step 215902  episodes 230  last return=-21  (agent 0 - opp 21)
  PPO step 221832  episodes 235  last return=-19  (agent 2 - opp 21)
  PPO step 227882  episodes 240  last return=-17  (agent 4 - opp 21)
  PPO step 234138  episodes 245  last return=-20  (agent 1 - opp 21)
  PPO step 240331  episodes 250  last return=-17  (agent 4 - opp 21)
  PPO update  policy_loss=-0.0036  value_loss=0.0444
  PPO step 247292  episodes 255  last return=-20  (agent 1 - opp 21)
  PPO update  policy_loss=-0.0033  value_loss=0.0477
  PPO step 254570  episodes 260  last return=-16  (agent 5 - opp 21)
  PPO step 261185  episodes 265  last return=-19  (agent 2 - opp 21)
  PPO update  policy_loss=-0.0050  value_loss=0.0338
  PPO step 267467  episodes 270  last return=-20  (agent 1 - opp 21)
  PPO update  policy_loss=-0.0037  value_loss=0.0537
  PPO step 274305  episodes 275  last return=-19  (agent 2 - opp 21)
  PPO update  policy_loss=-0.0053  value_loss=0.0416
  PPO step 281615  episodes 280  last return=-18  (agent 3 - opp 21)
  PPO update  policy_loss=-0.0045  value_loss=0.0547
  PPO step 289592  episodes 285  last return=-16  (agent 5 - opp 21)
  PPO update  policy_loss=-0.0037  value_loss=0.0426
  PPO step 297265  episodes 290  last return=-19  (agent 2 - opp 21)
  PPO step 304548  episodes 295  last return=-19  (agent 2 - opp 21)
  PPO update  policy_loss=-0.0048  value_loss=0.0364
  PPO step 312193  episodes 300  last return=-15  (agent 6 - opp 21)
  PPO update  policy_loss=-0.0032  value_loss=0.0399
  PPO step 320328  episodes 305  last return=-16  (agent 5 - opp 21)
  PPO update  policy_loss=-0.0048  value_loss=0.0419
  PPO step 329402  episodes 310  last return=-13  (agent 8 - opp 21)
  PPO update  policy_loss=-0.0039  value_loss=0.0502
  PPO step 338044  episodes 315  last return=-15  (agent 6 - opp 21)
  PPO update  policy_loss=-0.0046  value_loss=0.0492
  PPO step 347366  episodes 320  last return=-14  (agent 7 - opp 21)
  PPO update  policy_loss=-0.0059  value_loss=0.0408
  PPO step 356608  episodes 325  last return=-13  (agent 8 - opp 21)
  PPO update  policy_loss=-0.0049  value_loss=0.0426
  PPO step 366630  episodes 330  last return=-9  (agent 12 - opp 21)
  PPO step 376640  episodes 335  last return=-15  (agent 6 - opp 21)
  PPO update  policy_loss=-0.0048  value_loss=0.0404
  PPO step 387014  episodes 340  last return=-17  (agent 4 - opp 21)
  PPO update  policy_loss=-0.0059  value_loss=0.0261
  PPO update  policy_loss=-0.0058  value_loss=0.0404
  PPO step 399374  episodes 345  last return=-8  (agent 13 - opp 21)
  PPO step 411198  episodes 350  last return=-9  (agent 12 - opp 21)
  PPO update  policy_loss=-0.0056  value_loss=0.0303
  PPO step 422707  episodes 355  last return=-8  (agent 13 - opp 21)
  PPO update  policy_loss=-0.0054  value_loss=0.0332
  PPO step 434594  episodes 360  last return=-13  (agent 8 - opp 21)
  PPO update  policy_loss=-0.0050  value_loss=0.0415
  PPO step 446870  episodes 365  last return=-8  (agent 13 - opp 21)
  PPO update  policy_loss=-0.0047  value_loss=0.0328
  PPO step 459078  episodes 370  last return=-3  (agent 18 - opp 21)
  PPO update  policy_loss=-0.0056  value_loss=0.0335
  PPO step 470331  episodes 375  last return=-5  (agent 16 - opp 21)
  PPO update  policy_loss=-0.0052  value_loss=0.0292
  PPO step 481935  episodes 380  last return=-10  (agent 11 - opp 21)
  PPO update  policy_loss=-0.0041  value_loss=0.0375
  PPO step 494705  episodes 385  last return=-7  (agent 14 - opp 21)
  PPO update  policy_loss=-0.0018  value_loss=0.0333
  PPO step 509129  episodes 390  last return=-1  (agent 20 - opp 21)
  PPO update  policy_loss=-0.0043  value_loss=0.0289
  PPO step 523260  episodes 395  last return=3  (agent 21 - opp 18)
  PPO update  policy_loss=-0.0058  value_loss=0.0261
  PPO step 537881  episodes 400  last return=-2  (agent 19 - opp 21)
  PPO update  policy_loss=-0.0028  value_loss=0.0279
  PPO update  policy_loss=-0.0044  value_loss=0.0315
  PPO step 551767  episodes 405  last return=-3  (agent 18 - opp 21)
  PPO update  policy_loss=-0.0072  value_loss=0.0306
  PPO update  policy_loss=-0.0069  value_loss=0.0253
  PPO step 566448  episodes 410  last return=-3  (agent 18 - opp 21)
  PPO update  policy_loss=-0.0059  value_loss=0.0294
  PPO step 581337  episodes 415  last return=3  (agent 21 - opp 18)
  PPO update  policy_loss=-0.0039  value_loss=0.0289
  PPO update  policy_loss=-0.0061  value_loss=0.0198
  PPO step 596125  episodes 420  last return=-2  (agent 19 - opp 21)
  PPO update  policy_loss=-0.0078  value_loss=0.0163
  PPO step 610422  episodes 425  last return=4  (agent 21 - opp 17)
  PPO update  policy_loss=-0.0053  value_loss=0.0254
  PPO step 624117  episodes 430  last return=2  (agent 21 - opp 19)
  PPO update  policy_loss=-0.0051  value_loss=0.0193
  PPO update  policy_loss=-0.0066  value_loss=0.0199
  PPO step 638124  episodes 435  last return=4  (agent 21 - opp 17)
  PPO update  policy_loss=-0.0059  value_loss=0.0249
  PPO step 651944  episodes 440  last return=1  (agent 21 - opp 20)
  PPO update  policy_loss=-0.0084  value_loss=0.0247
  PPO step 666435  episodes 445  last return=4  (agent 21 - opp 17)
  PPO update  policy_loss=-0.0055  value_loss=0.0242
  PPO step 679793  episodes 450  last return=8  (agent 21 - opp 13)
  PPO update  policy_loss=-0.0072  value_loss=0.0182
  PPO update  policy_loss=-0.0043  value_loss=0.0214
  PPO step 692991  episodes 455  last return=8  (agent 21 - opp 13)
  PPO update  policy_loss=-0.0061  value_loss=0.0204
  PPO step 706418  episodes 460  last return=10  (agent 21 - opp 11)
  PPO update  policy_loss=-0.0047  value_loss=0.0189
  PPO update  policy_loss=-0.0085  value_loss=0.0257
  PPO step 719742  episodes 465  last return=9  (agent 21 - opp 12)
  PPO update  policy_loss=-0.0080  value_loss=0.0205
  PPO step 733775  episodes 470  last return=9  (agent 21 - opp 12)
  PPO update  policy_loss=-0.0051  value_loss=0.0133
  PPO step 747346  episodes 475  last return=8  (agent 21 - opp 13)
  PPO update  policy_loss=-0.0077  value_loss=0.0293
  PPO update  policy_loss=-0.0071  value_loss=0.0164
  PPO step 761101  episodes 480  last return=4  (agent 21 - opp 17)
  PPO update  policy_loss=-0.0060  value_loss=0.0248
  PPO step 774604  episodes 485  last return=11  (agent 21 - opp 10)
  PPO update  policy_loss=-0.0077  value_loss=0.0189
  PPO step 787189  episodes 490  last return=12  (agent 21 - opp 9)
  PPO update  policy_loss=-0.0094  value_loss=0.0142
  PPO step 800208  episodes 495  last return=15  (agent 21 - opp 6)
  PPO update  policy_loss=-0.0067  value_loss=0.0143
  PPO update  policy_loss=-0.0108  value_loss=0.0186
  PPO step 814077  episodes 500  last return=8  (agent 21 - opp 13)
  PPO update  policy_loss=-0.0105  value_loss=0.0173
  PPO step 828031  episodes 505  last return=12  (agent 21 - opp 9)
  PPO update  policy_loss=-0.0105  value_loss=0.0152
  PPO step 841653  episodes 510  last return=9  (agent 21 - opp 12)
  PPO update  policy_loss=-0.0068  value_loss=0.0165
  PPO update  policy_loss=-0.0082  value_loss=0.0217
  PPO step 855371  episodes 515  last return=7  (agent 21 - opp 14)
  PPO update  policy_loss=-0.0085  value_loss=0.0204
  PPO step 868935  episodes 520  last return=10  (agent 21 - opp 11)
  PPO update  policy_loss=-0.0116  value_loss=0.0130
  PPO step 882479  episodes 525  last return=11  (agent 21 - opp 10)
  PPO update  policy_loss=-0.0111  value_loss=0.0208
  PPO update  policy_loss=-0.0109  value_loss=0.0151
  PPO step 895988  episodes 530  last return=9  (agent 21 - opp 12)
  PPO update  policy_loss=-0.0131  value_loss=0.0158
  PPO step 909665  episodes 535  last return=5  (agent 21 - opp 16)
  PPO update  policy_loss=-0.0102  value_loss=0.0222
  PPO step 923510  episodes 540  last return=7  (agent 21 - opp 14)
  PPO update  policy_loss=-0.0101  value_loss=0.0208
  PPO update  policy_loss=-0.0121  value_loss=0.0189
  PPO step 937864  episodes 545  last return=6  (agent 21 - opp 15)
  PPO update  policy_loss=-0.0116  value_loss=0.0279
  PPO update  policy_loss=-0.0107  value_loss=0.0177
  PPO step 951928  episodes 550  last return=9  (agent 21 - opp 12)
  PPO update  policy_loss=-0.0100  value_loss=0.0145
  PPO update  policy_loss=-0.0114  value_loss=0.0184
  PPO step 965299  episodes 555  last return=11  (agent 21 - opp 10)
  PPO update  policy_loss=-0.0124  value_loss=0.0145
  PPO step 978971  episodes 560  last return=11  (agent 21 - opp 10)
  PPO update  policy_loss=-0.0129  value_loss=0.0263
  PPO step 993731  episodes 565  last return=9  (agent 21 - opp 12)
  PPO update  policy_loss=-0.0116  value_loss=0.0133
  PPO step 1008390  episodes 570  last return=15  (agent 21 - opp 6)
  PPO update  policy_loss=-0.0119  value_loss=0.0143
  PPO step 1022468  episodes 575  last return=11  (agent 21 - opp 10)
  PPO update  policy_loss=-0.0127  value_loss=0.0164
  PPO step 1037929  episodes 580  last return=17  (agent 21 - opp 4)
  PPO update  policy_loss=-0.0102  value_loss=0.0137
  PPO update  policy_loss=-0.0108  value_loss=0.0149
  PPO step 1052596  episodes 585  last return=11  (agent 21 - opp 10)
  PPO update  policy_loss=-0.0116  value_loss=0.0149
  PPO update  policy_loss=-0.0127  value_loss=0.0108
  PPO step 1067243  episodes 590  last return=15  (agent 21 - opp 6)
  PPO update  policy_loss=-0.0122  value_loss=0.0152
  PPO step 1082391  episodes 595  last return=18  (agent 21 - opp 3)
  PPO update  policy_loss=-0.0131  value_loss=0.0153
  PPO update  policy_loss=-0.0089  value_loss=0.0147
  PPO step 1097035  episodes 600  last return=13  (agent 21 - opp 8)
  PPO update  policy_loss=-0.0140  value_loss=0.0121
  PPO update  policy_loss=-0.0126  value_loss=0.0145
  PPO step 1112025  episodes 605  last return=13  (agent 21 - opp 8)
  PPO update  policy_loss=-0.0106  value_loss=0.0178
  PPO update  policy_loss=-0.0118  value_loss=0.0164
  PPO step 1126231  episodes 610  last return=11  (agent 21 - opp 10)
  PPO update  policy_loss=-0.0099  value_loss=0.0158
  PPO update  policy_loss=-0.0081  value_loss=0.0129
  PPO step 1139032  episodes 615  last return=15  (agent 21 - opp 6)
  PPO update  policy_loss=-0.0120  value_loss=0.0122
  PPO step 1151853  episodes 620  last return=13  (agent 21 - opp 8)
  PPO update  policy_loss=-0.0126  value_loss=0.0183
  PPO step 1163491  episodes 625  last return=16  (agent 21 - opp 5)
  PPO update  policy_loss=-0.0120  value_loss=0.0235
  PPO step 1174672  episodes 630  last return=19  (agent 21 - opp 2)
  PPO update  policy_loss=-0.0094  value_loss=0.0124
  PPO step 1184493  episodes 635  last return=17  (agent 21 - opp 4)
  PPO update  policy_loss=-0.0114  value_loss=0.0147
  PPO step 1193893  episodes 640  last return=16  (agent 21 - opp 5)
  PPO update  policy_loss=-0.0091  value_loss=0.0139
  PPO update  policy_loss=-0.0104  value_loss=0.0223
  PPO step 1204648  episodes 645  last return=15  (agent 21 - opp 6)
  PPO update  policy_loss=-0.0121  value_loss=0.0067
  PPO step 1214351  episodes 650  last return=16  (agent 21 - opp 5)
  PPO update  policy_loss=-0.0105  value_loss=0.0172
  PPO step 1224581  episodes 655  last return=17  (agent 21 - opp 4)
  PPO update  policy_loss=-0.0101  value_loss=0.0129
  PPO step 1234234  episodes 660  last return=20  (agent 21 - opp 1)
  PPO update  policy_loss=-0.0121  value_loss=0.0053
  PPO step 1243270  episodes 665  last return=19  (agent 21 - opp 2)
  PPO step 1252273  episodes 670  last return=19  (agent 21 - opp 2)
  PPO update  policy_loss=-0.0108  value_loss=0.0080
  PPO step 1261556  episodes 675  last return=20  (agent 21 - opp 1)
  PPO update  policy_loss=-0.0097  value_loss=0.0036
  PPO step 1270854  episodes 680  last return=20  (agent 21 - opp 1)
  PPO update  policy_loss=-0.0101  value_loss=0.0074
  PPO step 1280363  episodes 685  last return=18  (agent 21 - opp 3)
  PPO update  policy_loss=-0.0122  value_loss=0.0059
  PPO step 1289299  episodes 690  last return=20  (agent 21 - opp 1)
  PPO update  policy_loss=-0.0124  value_loss=0.0025
  PPO step 1298573  episodes 695  last return=19  (agent 21 - opp 2)
  PPO step 1307753  episodes 700  last return=19  (agent 21 - opp 2)
  PPO update  policy_loss=-0.0065  value_loss=0.0121
  PPO step 1317049  episodes 705  last return=19  (agent 21 - opp 2)
  PPO step 1326230  episodes 710  last return=20  (agent 21 - opp 1)
  PPO update  policy_loss=-0.0087  value_loss=0.0101
  PPO step 1335636  episodes 715  last return=18  (agent 21 - opp 3)
  PPO update  policy_loss=-0.0098  value_loss=0.0057
  PPO step 1345124  episodes 720  last return=18  (agent 21 - opp 3)
  PPO update  policy_loss=-0.0102  value_loss=0.0077
  PPO step 1354581  episodes 725  last return=19  (agent 21 - opp 2)
  PPO update  policy_loss=-0.0106  value_loss=0.0037
  PPO step 1364182  episodes 730  last return=18  (agent 21 - opp 3)
  PPO update  policy_loss=-0.0105  value_loss=0.0050
  PPO step 1373694  episodes 735  last return=19  (agent 21 - opp 2)
  PPO update  policy_loss=-0.0114  value_loss=0.0039
  PPO step 1382781  episodes 740  last return=19  (agent 21 - opp 2)
  PPO update  policy_loss=-0.0108  value_loss=0.0062
  PPO step 1391840  episodes 745  last return=18  (agent 21 - opp 3)
  PPO update  policy_loss=-0.0135  value_loss=0.0073
  PPO step 1401305  episodes 750  last return=18  (agent 21 - opp 3)
  PPO update  policy_loss=-0.0113  value_loss=0.0036
  PPO step 1410540  episodes 755  last return=20  (agent 21 - opp 1)
  PPO update  policy_loss=-0.0112  value_loss=0.0046
  PPO step 1420317  episodes 760  last return=19  (agent 21 - opp 2)
  PPO update  policy_loss=-0.0092  value_loss=0.0098
  PPO step 1429514  episodes 765  last return=20  (agent 21 - opp 1)
  PPO step 1439197  episodes 770  last return=20  (agent 21 - opp 1)
  PPO update  policy_loss=-0.0114  value_loss=0.0065
  PPO step 1448341  episodes 775  last return=19  (agent 21 - opp 2)
  PPO update  policy_loss=-0.0120  value_loss=0.0021
  PPO step 1457400  episodes 780  last return=20  (agent 21 - opp 1)
  PPO update  policy_loss=-0.0122  value_loss=0.0017
  PPO step 1466546  episodes 785  last return=20  (agent 21 - opp 1)
  PPO step 1475639  episodes 790  last return=20  (agent 21 - opp 1)
  PPO update  policy_loss=-0.0088  value_loss=0.0101
  PPO step 1485042  episodes 795  last return=20  (agent 21 - opp 1)
  PPO update  policy_loss=-0.0137  value_loss=0.0178
  PPO step 1494639  episodes 800  last return=20  (agent 21 - opp 1)
  PPO update  policy_loss=-0.0107  value_loss=0.0028
    """
    
    # Parse the logs
    ppo_steps, episodes, last_returns, policy_losses, value_losses, value_losses_epochs = parse_ppo_logs(log_string)
    
    # Print summary statistics
    print("Parsed Data Summary:")
    print(f"Total episodes recorded: {len(episodes)}")
    print(f"Total updates recorded: {len(policy_losses)}")
    print(f"Episode range: {min(episodes)} - {max(episodes)}")
    print(f"Return range: {min(last_returns):.1f} - {max(last_returns):.1f}")
    print(f"Policy loss range: {min(policy_losses):.4f} - {max(policy_losses):.4f}")
    print(f"Value loss range: {min(value_losses):.4f} - {max(value_losses):.4f}")
    print()
    
    # Create the plot
    plot_training_results(
        episodes=episodes,
        last_returns=last_returns,
        policy_losses=policy_losses,
        value_losses=value_losses,
        value_losses_epochs=value_losses_epochs,
        title="PPO Training on Pong Environment",
        save_path="ppo_training_results.png"  # Optional: save to file
    )
    
    # You can also access the parsed data for further analysis
    print("\nFirst few parsed entries:")
    for i in range(min(3, len(episodes))):
        print(f"Episode {episodes[i]}: Return={last_returns[i]}")



if __name__ == "__main__":
    main()