from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load association rules only
association_rules = joblib.load("association_rules.pkl")

app = FastAPI()

class ProductRequest(BaseModel):
    name: str
    category: str = None  # Optional if you plan to use it later

@app.post("/suggest")
def suggest_associated_products(request: ProductRequest):
    associated_products = []

    for _, row in association_rules.iterrows():
        antecedents = [item.strip() for item in row['antecedents'].split(',')]
        if request.name in antecedents:
            consequents = [item.strip() for item in row['consequents'].split(',')]
            associated_products.extend(consequents)

    # Remove duplicates and exclude the input product
    associated_products = list(set(associated_products) - {request.name})

    return {
        "associatedProducts": associated_products
    }
