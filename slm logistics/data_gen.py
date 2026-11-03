"""
Logistics SLM — Layer 1: Data Generator.

Generates synthetic agent trajectories for 8 comprehensive logistics skills:

  Skill A — Shipment Tracking (tracking_agent)
  Skill B — Inventory Check (inventory_agent)
  Skill C — Route Alert (routing_agent)
  Skill D — Freight Cost Estimation (cost_agent)
  Skill E — Return Management (returns_agent)
  Skill F — Supplier Analytics (supplier_agent)
  Skill G — Warehouse Capacity (capacity_agent)
  Skill H — Document OCR (ocr_agent)
"""
import json
import random
import os

# Ensure reproducible dataset
random.seed(42)

# --- Base Entities ---
SHIPMENT_IDS = [f"SHP{str(i).zfill(4)}" for i in range(1000, 1050)]
LOCATIONS = ["Chicago Warehouse", "NY Hub", "LA Port", "Dallas Transit Center", "Miami Dock"]
ITEMS = ["forklift parts", "packaging tape", "pallets", "safety gear", "conveyor belts"]
ROUTES = ["NY-BOS", "CHI-DAL", "LA-SF", "MIA-ATL"]
SUPPLIERS = ["GlobalPack", "HeavyLift Inc", "SafeTransit Co", "QuickShip LLC"]
DOCUMENTS = ["shipping label", "commercial invoice", "bill of lading", "customs declaration"]

STATUSES = ["in transit", "delayed due to weather", "delivered", "held at customs", "out for delivery"]
STOCK_LEVELS = ["high (500+ units)", "medium (100-500 units)", "low (<100 units)", "out of stock"]
TRAFFIC_CONDS = ["clear skies and no traffic", "heavy traffic delays", "severe weather warning", "accident reported, expect delays"]
COST_ESTIMATES = ["$450.00", "$1,200.00", "$320.50", "$2,100.00"]
PERFORMANCE_METRICS = ["98% on-time", "85% on-time (needs improvement)", "92% on-time"]
CAPACITY_LEVELS = ["80% full", "20% full (plenty of space)", "100% full (at capacity)"]
EXTRACTED_DATA = ["total_weight: 450kg", "invoice_amount: $1200", "destination: LA Port", "sender: GlobalPack"]

def _format_chatml(user_msg: str, assistant_msg: str) -> str:
    return (
        f"<|im_start|> user\n{user_msg}<|im_end|>\n"
        f"<|im_start|> assistant\n{assistant_msg}<end>"
    )

# ── Skill A: Shipment Tracking ───────────────────────────────────────────────
def gen_tracking_example() -> str:
    shipment = random.choice(SHIPMENT_IDS)
    loc = random.choice(LOCATIONS)
    status = random.choice(STATUSES)
    user_msg = random.choice([f"what is the status of shipment {shipment} ?", f"track {shipment} for me"])
    asst_msg = f"<thought> user wants to track a shipment <route> tracking_agent <tool_call> get_shipment_status <observation> {status} | {loc} <action> Shipment {shipment} is currently {status} at {loc}."
    return _format_chatml(user_msg, asst_msg)

# ── Skill B: Inventory Check ────────────────────────────────────────────────
def gen_inventory_example() -> str:
    item = random.choice(ITEMS)
    loc = random.choice(LOCATIONS)
    stock = random.choice(STOCK_LEVELS)
    user_msg = random.choice([f"check inventory for {item} in {loc}", f"do we have any {item} left at the {loc}?"])
    asst_msg = f"<thought> user wants to check stock levels <route> inventory_agent <tool_call> check_stock <observation> {stock} <action> The inventory for {item} at {loc} is currently {stock}."
    return _format_chatml(user_msg, asst_msg)

# ── Skill C: Route Alert ─────────────────────────────────────────────────
def gen_routing_example() -> str:
    route = random.choice(ROUTES)
    cond = random.choice(TRAFFIC_CONDS)
    user_msg = random.choice([f"are there any alerts for route {route} ?", f"how is the traffic on {route} today?"])
    asst_msg = f"<thought> user wants route conditions <route> routing_agent <tool_call> get_route_status <observation> {cond} <action> The current status for route {route} is: {cond}."
    return _format_chatml(user_msg, asst_msg)

# ── Skill D: Freight Cost Estimation ──────────────────────────────────────
def gen_cost_example() -> str:
    item = random.choice(ITEMS)
    route = random.choice(ROUTES)
    cost = random.choice(COST_ESTIMATES)
    user_msg = random.choice([f"estimate freight cost for {item} on {route}", f"how much to ship {item} via {route}?"])
    asst_msg = f"<thought> user wants freight cost estimation <route> cost_agent <tool_call> get_freight_estimate <observation> {cost} <action> The estimated freight cost to ship {item} on route {route} is {cost}."
    return _format_chatml(user_msg, asst_msg)

# ── Skill E: Return Management ────────────────────────────────────────────
def gen_returns_example() -> str:
    shipment = random.choice(SHIPMENT_IDS)
    user_msg = random.choice([f"process a return for damaged goods on {shipment}", f"initiate return for {shipment}"])
    asst_msg = f"<thought> user wants to process a return <route> returns_agent <tool_call> initiate_return <observation> return_auth_generated <action> Return authorization for {shipment} has been successfully generated."
    return _format_chatml(user_msg, asst_msg)

# ── Skill F: Supplier Analytics ───────────────────────────────────────────
def gen_supplier_example() -> str:
    supplier = random.choice(SUPPLIERS)
    metric = random.choice(PERFORMANCE_METRICS)
    user_msg = random.choice([f"what is the on-time delivery rate for {supplier}?", f"supplier performance report for {supplier}"])
    asst_msg = f"<thought> user wants supplier analytics <route> supplier_agent <tool_call> get_supplier_metrics <observation> {metric} <action> The performance metric for {supplier} is: {metric}."
    return _format_chatml(user_msg, asst_msg)

# ── Skill G: Warehouse Capacity ───────────────────────────────────────────
def gen_capacity_example() -> str:
    loc = random.choice(LOCATIONS)
    cap = random.choice(CAPACITY_LEVELS)
    user_msg = random.choice([f"do we have space in {loc}?", f"what is the current capacity of {loc}?"])
    asst_msg = f"<thought> user wants warehouse capacity <route> capacity_agent <tool_call> check_capacity <observation> {cap} <action> The {loc} is currently {cap}."
    return _format_chatml(user_msg, asst_msg)

# ── Skill H: Document OCR ───────────────────────────────────────────────
def gen_ocr_example() -> str:
    doc = random.choice(DOCUMENTS)
    shipment = random.choice(SHIPMENT_IDS)
    data = random.choice(EXTRACTED_DATA)
    user_msg = random.choice([f"scan this {doc} for {shipment}", f"extract data from {shipment} {doc}"])
    asst_msg = f"<thought> user wants to extract text from document <route> ocr_agent <tool_call> process_document <observation> {data} <action> The extracted data from the {doc} is: {data}."
    return _format_chatml(user_msg, asst_msg)


GENERATORS = [
    gen_tracking_example, 
    gen_inventory_example, 
    gen_routing_example,
    gen_cost_example,
    gen_returns_example,
    gen_supplier_example,
    gen_capacity_example,
    gen_ocr_example
]

def generate_dataset(n_examples: int, output_file: str):
    examples = [random.choice(GENERATORS)() for _ in range(n_examples)]
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(ex + "\n")
            
    print(f"Generated {n_examples} logistics examples across {len(GENERATORS)} skills and saved to {output_file}")

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "logistics_dataset.txt")
    generate_dataset(15000, output_path)
