import pandas as pd

# Define the financial topics and their descriptions
categories = {
    "Category": [
        "Market Analysis",
        "Investment Strategies",
        "Financial News",
        "Personal Finance",
        "Corporate Finance",
        "Economic Reports",
        "Risk Management",
        "Financial Regulations",
        "Taxation",
        "Real Estate Finance",
        "Cryptocurrency",
        "Insurance",
        "Wealth Management",
        "Behavioral Finance",
        "Fintech Innovations"
    ],
    "Description": [
        "Analysis of stock market trends and economic indicators.",
        "Approaches to investing, including diversification and asset allocation.",
        "Current events and updates in the financial world.",
        "Managing personal finances, including budgeting and saving.",
        "Financial management practices within corporations.",
        "Reports on economic performance and forecasts.",
        "Strategies to identify and mitigate financial risks.",
        "Laws and regulations governing financial institutions.",
        "Understanding taxation and its implications for individuals.",
        "Finance related to buying, selling, and managing real estate.",
        "Digital currencies and blockchain technology.",
        "Financial products that provide protection against risks.",
        "Managing and growing individual wealth.",
        "Psychological influences on financial decisions.",
        "Innovations in financial technology and services."
    ]
}

# Create a DataFrame
df_categories = pd.DataFrame(categories)

# Save to CSV
csv_file_path = 'categories.csv'
df_categories.to_csv(csv_file_path, index=False)

print(f"Created '{csv_file_path}' with financial categories and descriptions.")
