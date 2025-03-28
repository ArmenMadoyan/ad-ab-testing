# 🧠 A/B Testing with Multi-Armed Bandits

This project demonstrates a simulation-based comparison of **Epsilon-Greedy** and **Thompson Sampling** algorithms on a 2-armed bandit problem. The goal is to optimize ad selection using reinforcement learning techniques, monitor the learning process, and visualize the results.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ArmenMadoyan/ad-ab-testing.git
cd ad-ab-testing

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the experiment
python Bandit.py

## Project Structure

ad-ab-testing/
├── Bandit.py              # Main script with algorithm implementations and visualizations
├── data/
│   └── # Stores (Bandit, Reward, Algorithm) results from experiments
├── img/
├── report/
│   └──  # LaTeX report template with results and figures
├── requirements.txt       # Python dependencies
└── README.md              # You are here 📌
