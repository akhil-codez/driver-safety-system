# Live Code Demonstration Guide
**Focus: Preprocessing & Data Simulation**

Use this script during your presentation to show the working code.

## 1. Setup (Before you start)
*   Open VS Code to your project folder.
*   Open the terminal (`Ctrl+` `).
*   Open two files in the editor so they are ready to show:
    1.  `python/simulate.py`
    2.  `visualize_data.py`

## 2. The Walkthrough

### Step A: Show the Logic (`simulate.py`)
*   **Say:** "To train our model effectively, we first needed valid driving data. Since gathering thousands of hours of dangerous driving data is risky, we built a physics-based simulator."
*   **Show Code:** Scroll to `generate_segment` function (around line 11).
*   **Explain:**
    *   "Here we simulate a vehicle's physics. We calculate **Centripetal Acceleration** (`acc_y`) using the formula `v^2 / r`."
    *   "We also add Gaussian noise (lines 65-71) to mimic real smartphone sensors, which are never perfect."

### Step B: Run the Simulation
*   **Say:** "I will now generate a synthetic trip with 10 minutes of driving data."
*   **Action:** Run this command in the terminal:
    ```bash
    python python/simulate.py --out data/live_test.csv
    ```
*   **Show Output:** Instead of opening the raw file, verify the data is valid by running:
    ```bash
    python python/show_data.py
    ```
    *This will print a clean table of the first 10 rows (Speed, G-Force, Location).*

### Step C: Visualizing the Data
*   **Say:** "Raw numbers are hard to interpret, so we built a visualization tool to verify our 'curves' look correct."
*   **Action:** Run:
    ```bash
    python visualize_data.py
    ```
*   **Show Result:** Open the generated `data_preview.png`.
*   **Explain:** "As you can see, when the vehicle enters a curve (top graph), our simulated gyroscope and accelerometer spike (bottom graph). This confirms our preprocessing logic captures the physical events correctly."

## 3. Conclusion of Demo
*   **Say:** "This validated dataset is now ready to be fed into our Neural Network for training, which is the next phase of our project."
