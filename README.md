# Equity-Assistant

# Portfolio Assistant 💬

A comprehensive web-based portfolio management tool for investors to analyze, manage, and optimize their stock portfolios. Built with **Streamlit**, this application supports multiple stock markets (India, US, UK) and provides detailed insights, including industry distributions, risk metrics, and optimized portfolio suggestions.

---

## Features 📊

- **Portfolio Analysis:**
  - Display market cap classification (Small, Mid, Large).
  - Industry and market cap distribution charts.
  - Allocation breakdown for selected stocks.
  
- **Performance Metrics:**
  - Total Portfolio Value and Returns.
  - Daily Gain/Loss (absolute and percentage).
  
- **Risk Metrics:**
  - Volatility (Standard Deviation).
  - Sharpe Ratio for evaluating risk-adjusted returns.
  - Value at Risk (VaR) and Conditional VaR (CVaR) calculations.

- **Optimization:**
  - Portfolio optimization to maximize Sharpe Ratio.
  - Suggested weights and number of shares for each stock.

- **Enhanced Visualization:**
  - Pie charts for distribution analysis.
  - Custom circular logos for companies.

---

## Installation 🛠️

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ekalavya-portfolio-assistant.git
   cd ekalavya-portfolio-assistant
   ```

2. **Install dependencies:**
   Create and activate a virtual environment (optional but recommended).
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the necessary files:**
   - Place the stock data files in the `downloads/python` directory.
     - `USStocks.xlsx`
     - `UKStocks.xlsx`
     - `INDStocks.xlsx`

4. **Run the application:**
   ```bash
   streamlit run main.py
   ```

---

## File Structure 📁

- `main.py`: The main application script.
- `USStocks.xlsx`, `UKStocks.xlsx`, `INDStocks.xlsx`: Stock data files for different regions.
- `requirements.txt`: Required Python dependencies.

---

## Key Libraries Used 📚

- **Streamlit**: For building the interactive web interface.
- **yfinance**: To fetch stock data and historical prices.
- **Pandas**: For data manipulation.
- **Matplotlib**: For creating distribution charts.
- **NumPy**: For mathematical computations.
- **SciPy**: For optimization and statistical analysis.
- **Pillow**: To process and customize stock logos.

---

## Usage 💡

1. **Select the region:** Choose your preferred exchange (India, US, UK).
2. **Add companies:** Use the multiselect dropdown to pick stocks.
3. **Modify portfolio:** Specify the number of shares for each stock.
4. **View analytics:**
   - Check portfolio allocation and performance summaries.
   - Analyze risk metrics and optimization results.
5. **Download optimized weights:** Review the suggested portfolio with calculated metrics.

---

## Screenshots 🖼️

![Portfolio Overview](https://github.com/EkalavyaPrasad/Equity-Assistant/blob/d9485abfc2221ccfee67c5098ac9917417d6604d/EPA%20SS1.png)
![Risk Metrics](path/to/screenshot2.png)
![Optimized Portfolio](path/to/screenshot3.png)

---

## Future Improvements 🚀

- Support for additional stock exchanges.
- Real-time data integration with APIs.
- Enhanced visualizations with interactive charts.
- Integration with brokerage accounts for live portfolio tracking.

---

## Contributing 🤝

Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request.

---

## Contact 📬

If you have any questions or suggestions, feel free to reach out:
- **LinkedIn**: www.linkedin.com/in/ekalavya-prasad
- **Email**: eklavya.prasad2709@gmail.com
- **GitHub**: https://github.com/EkalavyaPrasad

---

### Notes:
- Replace placeholders like `your-username`, `your-linkedin-profile`, and `your-email@example.com` with your actual details.
- Add screenshots in the appropriate section and update their file paths.
- Ensure the `LICENSE` file is in your project directory if you reference it.
