import matplotlib.pyplot as plt
import pandas as pd
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot losses from a JSON file')
    parser.add_argument('--json', type=str, default="losses.json", help='Path to the JSON file containing the losses')
    args = parser.parse_args()
    json_file = args.json
    
    # Load the JSON file
    with open(json_file, "r") as f:
        losses = json.load(f)
        
    # Convert the losses to a DataFrame
    df = pd.DataFrame(losses)
    
    
    # Plot the losses
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df[column], label=column)
    
    # Plot a line over 0
    plt.axhline(0, color='green', linewidth=0.5)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    
    plt.show()
    
    
