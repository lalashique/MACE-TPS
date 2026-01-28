

from multiprocessing import Process

from mlp_path.tps import TPS

# Main entry point for the script
if __name__ == "__main__":
    # Parse command-line arguments
    # python run.py -dir "." -cfg "config.json" -nr 1000 -traj "p1.xyz"
    tps = TPS()