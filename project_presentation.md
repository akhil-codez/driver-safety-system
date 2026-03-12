# Smart Road Early Warning System
**A Machine Learning Approach to Road Safety**

---

**Guided By:**
Mrs. Neethu K
*Asst. Professor, Dept. of AI & DS*

**Group Members:**
*   Muhammed Shibil K P (SPT23AD034)
*   Shehruz Muhammed C S (SPT23AD043)
*   Adithya Chandran (SPT23AD006)
*   Akhil P (SPT23AD009)

---

## Contents
1.  Introduction
2.  Motivation
3.  Objectives
4.  Scope & Significance
5.  Literature Review
6.  Block Diagram
7.  Methodology
8.  Technical Feasibility
9.  Components & Tech Stack
10. Estimated Cost
11. Work Progress
12. Future Scope
13. Application Gallery
14. Bibliography

<div style="page-break-after: always;"></div>

## 1. Introduction
The **Smart Road Early Warning System** is an innovative mobile application designed to enhance driver safety on hazardous road segments.

*   **Core Function**: Detects sharp curves and road anomalies in real-time.
*   **Technology**: Utilizes standard smartphone sensors (GPS, Accelerometer, Gyroscope) combined with Machine Learning.
*   **Interaction**: Provides immediate haptic (vibration) and visual alerts to the driver *before* they enter a danger zone.
*   **Goal**: To reduce accidents caused by speeding on unfamiliar or curved roads.

## 2. Motivation
*   **High Accident Rates**: Curves and hilly terrains are hotspots for vehicle rollovers and loss-of-control accidents.
*   **Cost Factor**: Traditional Advanced Driver Assistance Systems (ADAS) require expensive, dedicated hardware (LiDAR, Radar).
*   **Accessibility**: Almost every driver owns a smartphone, making this a zero-cost safety upgrade for the masses.
*   **Data-Driven**: Leveraging AI to learn from driving patterns rather than static rules improves accuracy over time.

## 3. Objectives
*   To develop a **cost-effective** driver warning system using readily available smartphone hardware.
*   To implement **Machine Learning algorithms** (TensorFlow Lite) on the edge for real-time curve severity classification.
*   To ensure **low-latency alerts** that give drivers sufficient reaction time.
*   To create a **distraction-free interface** that relies primarily on haptic feedback.

## 4. Scope & Significance
*   **Target Audience**: Daily commuters, truck fleet operators, and tourists navigating unfamiliar routes.
*   **Scalability**: The software-only approach allows for rapid deployment to millions of devices without manufacturing overhead.
*   **Infrastructure Diagnostic**: Aggregated data can help road authorities identify dangerous spots that need physical signage or repair.

<div style="page-break-after: always;"></div>

## 5. Literature Review (Part 1)

| Paper / Study | Key Insight |
| :--- | :--- |
| **1. Low-cost realtime horizontal curve detection (IEEE 2016)** | Used accelerometer/gyro to model curves with 93.8% accuracy, proving low-cost sensors are viable. |
| **2. Evaluation of curve speed warning system (CDC 2022)** | Found that dynamic warnings reduce curve entry speeds by 4-7 mph, significantly lowering rollover risk. |
| **3. Vibrotactile Patterns for ADAS (ResearchGate)** | Haptic feedback (vibration) resulted in 20% faster reaction times compared to visual-only alerts. |
| **4. Estimation of Road Curvature from GPS (Wang et al.)** | Proposed mathematical models to derive curve radius purely from raw GPS coordinates. |
| **5. Smartphone-based driver behavior profiling (IEEE Access)** | Used sensor fusion to classify driving styles (Aggressive vs. Safe) with 85% accuracy. |

<div style="page-break-after: always;"></div>

## Literature Review (Part 2)

| Paper / Study | Key Insight |
| :--- | :--- |
| **6. Multi-Sensor Fusion for Lane Detection (NIH)** | Combined camera and steering angle data to improve detection in sharp curves where vision fails. |
| **7. Vehicle Trajectory Estimation (ResearchGate)** | Used Kalman Filtering on smartphone data to predict future vehicle position and detect lane changes. |
| **8. Mobile Mapping Systems for Curve Identification** | Compared dedicated mapping hardware vs. smartphones; smartphones showed <5% error margin. |
| **9. Driving Style Recognition (Johnson & Trivedi)** | Analyzed dynamic time warping (DTW) on sensor streams to detect anomalous events like swerving. |

## 6. Block Diagram

![Block Diagram](block.png)

1.  **Input Layer**: Captures raw sensor data at 10Hz.
2.  **Processing Layer**: Computes features (speed, bearing change, lateral acceleration) and runs the Neural Network.
3.  **Output Layer**: Triggers vibration patterns based on severity (Mild, Urgent, Hectic).

<div style="page-break-after: always;"></div>

## 7. Methodology (App Working)
1.  **Trip Initiation**: Driver launches the app and inputs the destination (optional). The app begins monitoring in the background.
2.  **Sensor Data Acquisition**: The app continuously reads data from the **Accelerometer**, **Gyroscope**, and **GPS** sensors at a frequency of 10Hz.
3.  **Real-time Feature Extraction**: Raw sensor data is buffered into 3-second sliding windows. Key features (speed, bearing change, lateral g-force) are computed locally on the device.
4.  **On-Device Inference**: The extracted features are fed into the embedded **TensorFlow Lite model**, which classifies the current driving state (Safe, Mild Curve, Sharp Curve).
5.  **Alert Generation**: If a dangerous curve is detected:
    *   **Haptic Engine** triggers a specific vibration pattern.
    *   **Visual UI** flashes a warning color (Orange/Red) on the screen.
6.  **Feedback Loop**: The system continues to monitor for the next window, ensuring continuous protection throughout the drive.

<div style="page-break-after: always;"></div>

## 8. Data Simulation & Visualization (Current Progress)
As part of the **Preprocessing Stage**, we have successfully simulated driving scenarios to validate our feature extraction logic.

![Data Preview](data_preview.png)

*   **Top Graph**: Simulated GPS path showing both straight segments and sharp curves.
*   **Bottom Graph**: Corresponding sensor readings. Note the spikes in **Lateral Acceleration (Red)** and **Yaw Rate (Green)** during curve events, which our algorithm detects.

## 8. Technical Feasibility
*   **Hardware Agnostic**: Runs on any Android device with standard sensors (Android 8.0+).
*   **Edge Computing**: All processing happens locally on the phone. No internet connection is required for the core safety features, ensuring reliability in remote areas.
*   **Battery Efficiency**: Optimized sensor polling and lightweight model inference minimize power drain.

## 9. Software Components & Tech Stack
*   **Development Environments (IDE)**:
    *   **Android Studio**: Primary IDE for native Android app development.
    *   **VS Code**: Used for Python data analysis and script writing.
    *   **Jupyter Notebook**: For interactive data visualization and model prototyping.
*   **Languages**:
    *   **Kotlin**: For the Android application logic.
    *   **Python**: For data simulation, preprocessing, and model training.
*   **Key Libraries**:
    *   **TensorFlow Lite**: For running ML models efficiently on the phone.
    *   **Pandas & NumPy**: For data manipulation and math operations.
    *   **Scikit-Learn**: For feature scaling and pre-model evaluation.
    *   **Matplotlib**: For generating data graphs and visualizations.
    *   **Google Maps SDK**: For displaying routes and alerts on the map.
*   **Tools**:
    *   **Git & GitHub**: For version control and collaboration.
    *   **xhtml2pdf**: For generating project documentation and reports.

<div style="page-break-after: always;"></div>

## 10. Estimated Cost
Our solution is designed to be **Zero-Cost** for the end user and highly affordable for development.

| Component | Cost (INR) | Reasoning |
| :--- | :--- | :--- |
| **Hardware** | ₹0 | Utilizes the driver's existing Android Smartphone. |
| **Software License** | ₹0 | Built using Open Source technologies (Kotlin, Python, TensorFlow). |
| **Cloud/Server** | ₹0 | **Edge Computing** architecture means no recurring server costs. |
| **Map Data** | ₹0 | Google Maps SDK (Free tier for development). |
| **Total** | **₹0** | **A completely free-to-use safety upgrade.** |

<div style="page-break-after: always;"></div>

## 11. Work Progress (Status: Preprocessing Complete)
*   [x] **Phase 1: Research (Completed)**
    *   Literature survey conducted (9+ papers).
    *   Key features for curve detection identified (Speed, Yaw Rate, Centripetal Accel).
*   [x] **Phase 2: Data Preprocessing (Completed)**
    *   **Simulation Engine**: Developed `simulate.py` to generate synthetic driving traces.
    *   **Labeling Logic**: Rule-based curvature calculation implemented.
    *   **Analysis**: Sensor noise modeling and filtering applied.

## 12. Bibliography
1.  *Low-cost realtime horizontal curve detection using inertial sensors of a smartphone*. IEEE VTC 2016.
2.  Simeonov et al., 2022. *Evaluation of advanced curve speed warning system*.
3.  *Vibrotactile Patterns for Smartphone-Based ADAS Warnings*. ResearchGate.
4.  *Estimation of Road Centerline Curvature from Raw GPS Data*. ResearchGate.
5.  *Smartphone-based driver behavior analysis and profiling*. IEEE Access.
6.  *Multi-Sensor Fusion for Lane Detection in Curves*. NIH.
7.  *Vehicle Trajectory Estimation using Smartphone Sensors*. ResearchGate.
8.  *Mobile Mapping Systems (MMS) for Curve Identification*.
9.  Johnson & Trivedi. *Driving Style Recognition Using a Smartphone as a Sensor Platform*.
10. Documentation: *TensorFlow Lite Guide*, *Android Developer Docs*.

---
*Created for ADD 334 - Mini Project Zeroth Review*
