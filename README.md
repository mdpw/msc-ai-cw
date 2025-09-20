# AI-Powered Cinnamon Grading System

The grading process of cinnamon products remains inconsistent, which creates problems for market pricing and supply chain management and product distribution. The manual grading process remains both time-consuming and inconsistent while being subjective which results in supply chain mismatches and revenue losses and product waste. A computerized system needs to exist for Ceylon cinnamon quality classification into High, Medium and Low grades through chemical composition analysis. A predictive machine learning model can identify complex nonlinear connections between chemical properties and quality grades by analyzing twelve essential parameters which include Moisture, Ash, Volatile Oil, Acid Insoluble Ash, Chromium, Coumarin etc. The model provides quick and uniform quality evaluations at scale which helps organizations maintain quality standards and set prices and meet export requirements and optimize their supply chain operations.

Business Need
Current manual quality testing is slow, subjective, and inconsistent. By applying ANN:
•	Quality grading becomes automated and standardized, which helps in improving the Brand Trust.
•	Businesses gain data-driven insights for better pricing, quality control, and regulatory compliance.
•	Decision-making is faster, enabling efficient processing and shipment planning.
•	Risk Reduction & Compliance
•	Product Development & Innovation



Steps to Run Commands

## To train the ANN base model 
1. Go to root folder
2. Execute command: python -m main_app.training.train_base_model  

## To train the ANN best model 
1. Go to root folder
2. Execute command: python -m main_app.training.train_best_model 

## To start Cinnamon Quiality App
1. Go to root folder
2. Open terminal 1
3. Execute command: python -m main_app.app (This will open Main app in port 8001)
4. Open terminal 2
5. Execute command: python xai_app/app.py (This will open XAI app in port 8002)
6. Open terminal 3
7. Execute command: python genai_app/app.py (This will open GenAI app in port 8003)
8. Open terminal 4
9. Execute command: python app.py (This will open App with the home page in port 5000)
10. Go the browser and browse http://127.0.0.1:5000