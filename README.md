# Equity-Assistant

## Portfolio Assistant 💬

A comprehensive web-based portfolio management tool for investors to analyze, manage, and optimize their stock portfolios. Built with **Streamlit**, this application supports multiple stock markets (India, US, UK) and provides detailed insights, including industry distributions, risk metrics, and optimized portfolio suggestions.

---

## Features 📊

### Portfolio Analysis  
- Market cap classification: Small, Mid, Large.  
- Industry and market cap distribution charts.  
- Allocation breakdown for selected stocks.  

### Performance Metrics  
- Total Portfolio Value and Returns.  
- Daily Gain/Loss (absolute and percentage).  

### Risk Metrics  
- Volatility (Standard Deviation).  
- Sharpe Ratio for evaluating risk-adjusted returns.  
- Value at Risk (VaR) and Conditional VaR (CVaR).  

### Optimization  
- Portfolio optimization to maximize the Sharpe Ratio.  
- Suggested weights and number of shares for each stock.  

### AI-Powered Insights  
- **Ollama Integration**: Generate AI-based reports and insights for your portfolio.  

### Enhanced Visualization  
- Pie charts for distribution analysis.  
- Custom circular logos for companies.  

---

## Installation 🛠️

### Prerequisites  
Ensure the following are installed on your system:  
- **Python** 3.8 or higher  
- **pip** (Python package manager)  
- **Git** (optional, for cloning the repository)  
- **Ollama**: To enable AI-based insights (see instructions below).  

### Steps  

1. **Clone the Repository:**  
   Clone the repository using Git or download the ZIP file:  
   ```bash
   git clone https://github.com/EkalavyaPrasad/Equity-Assistant.git
   cd Equity-Assistant
   ```  

   If you downloaded the ZIP file, extract it and navigate to the extracted directory.  

2. **Set Up a Virtual Environment (Optional but Recommended):**  
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```  

3. **Install Dependencies:**  
   Install all required Python libraries using `requirements.txt`:  
   ```bash
   pip install -r requirements.txt
   ```  

4. **Set Up Ollama:**  
   - **Install Ollama**:  
     Ollama is required for generating AI-based portfolio reports. Follow the [official Ollama installation guide](https://ollama.com/download) to set it up.  
   - **Download Models**: Ensure you have downloaded the required models (like `llama3.3` or others). Use the following command in your terminal after installing Ollama:  
     ```bash
     ollama pull llama3.3
     ```  

5. **Add Stock Data Files:**  
   Place the following files in the `downloads/python` directory:  
   - `USStocks.xlsx`  
   - `UKStocks.xlsx`  
   - `INDStocks.xlsx`  

6. **Run the Application:**  
   Start the application using Streamlit:  
   ```bash
   streamlit run main.py
   ```  

---

## File Structure 📁  

- `main.py`: Main application script.  
- `downloads/python/`: Directory to store stock data files.  
  - `USStocks.xlsx`, `UKStocks.xlsx`, `INDStocks.xlsx`: Stock data for respective regions.  
- `requirements.txt`: List of Python dependencies.  

---

## Key Libraries and Tools Used 📚  

- **Streamlit**: Interactive web application framework.  
- **yfinance**: Fetch stock data and historical prices.  
- **Pandas**: Data manipulation and analysis.  
- **Matplotlib**: Visualization and charts.  
- **NumPy**: Mathematical computations.  
- **SciPy**: Optimization and statistical analysis.  
- **Pillow**: Processing and customizing stock logos.  
- **Ollama**: AI-powered insights and report generation.  

---

## Usage 💡  

1. **Select the Region:** Choose a stock exchange (India, US, UK).  
2. **Add Companies:** Use the dropdown to pick stocks.  
3. **Modify Portfolio:** Specify the number of shares for each stock.  
4. **View Analytics:**  
   - Check portfolio allocation and performance summaries.  
   - Analyze risk metrics and optimization results.  
5. **Generate AI Reports:**  
   - Use the built-in Ollama model to generate insights and reports.  

---

## Screenshots 🖼️  

### Portfolio Overview  
![Portfolio Overview](https://github.com/EkalavyaPrasad/Equity-Assistant/blob/d9485abfc2221ccfee67c5098ac9917417d6604d/EPA%20SS1.png)  

### Portfolio Dataframe  
![Portfolio_df](https://github.com/EkalavyaPrasad/Equity-Assistant/blob/d9485abfc2221ccfee67c5098ac9917417d6604d/EPA%20SS2.png)  

### Portfolio Summary  
![Portfolio Summary](https://github.com/EkalavyaPrasad/Equity-Assistant/blob/d9485abfc2221ccfee67c5098ac9917417d6604d/EPA%20SS3.png)  

### Portfolio Optimization  
![Portfolio Optimization](https://github.com/EkalavyaPrasad/Equity-Assistant/blob/d9485abfc2221ccfee67c5098ac9917417d6604d/EPA%20SS4.png)  

### AI-Generated Report  
![AI Report Generation](https://github.com/EkalavyaPrasad/Equity-Assistant/blob/d9485abfc2221ccfee67c5098ac9917417d6604d/EPA%20SS5.png)  

---

## Future Improvements 🚀  

- Support for additional stock exchanges.  
- Real-time data integration with APIs.  
- Enhanced visualizations with interactive charts.  
- Integration with brokerage accounts for live portfolio tracking.  
- Fixing UK stock ticker issues.  
- Improved error handling.  

---

## Contributing 🤝  

Contributions are welcome! To contribute:  
1. Fork the repository.  
2. Create a feature branch.  
3. Commit your changes.  
4. Submit a pull request.  

---

## Contact 📬  

Feel free to reach out with questions or suggestions:  
- **LinkedIn**: https://www.linkedin.com/in/ekalavya-prasad
- **Email**: [eklavya.prasad2709@gmail.com](mailto:eklavya.prasad2709@gmail.com)  
- **GitHub**: [Ekalavya Prasad](https://github.com/EkalavyaPrasad)
- **portfolio**: https://www.ekalavyaprasad.com
- **Read about this project on my Blog**: https://bsidianblog.vercel.app/
