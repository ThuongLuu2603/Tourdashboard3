"""
Data generator for Vietravel Business Intelligence Dashboard
Generates realistic mock data for tour sales, customers, and operations

Clean data generator for Vietravel dashboard.

Provides:
- VietravelDataGenerator: mock data generators
- load_or_generate_data(spreadsheet_url=None): loads a public Google Sheet (CSV export)
  mapping columns E,F,G,I,J,P,Q,R,S into the tours dataset. Falls back to mock data.
"""

import io
import random
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from faker import Faker

# Initialize Faker with Vietnamese locale
fake = Faker(['vi_VN'])


class VietravelDataGenerator:
    """Generates realistic mock data for Vietravel tour business"""

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)

        # Small sets used for realistic generation
        self.tour_routes = [
            "DH & ĐBSH", "Nam Trung Bộ", "Bắc Trung Bộ", "Liên Tuyến miền Tây",
            "Phú Quốc", "Thái Lan", "Trung Quốc", "Hàn Quốc", "Singapore - Malaysia",
            "Nhật Bản", "Châu Âu", "Châu Mỹ", "Châu Úc", "Châu Phi", "Tây Bắc",
            "Đông Bắc", "Tây Nguyên"
        ]

        self.business_units = ["Miền Trung", "Miền Tây", "Miền Bắc", "Trụ sở & ĐNB"]
        self.sales_channels = ["Online", "Trực tiếp VPGD", "Đại lý"]
        self.segments = ["FIT", "GIT", "Inbound"]

        self.partners = [
            ("Khách sạn A", "Khách sạn"), ("Khách sạn B", "Khách sạn"), ("Khách sạn C", "Khách sạn"),
            ("Hàng không X", "Vé máy bay"), ("Hàng không Y", "Vé máy bay"),
            ("Vận chuyển 1", "Vận chuyển"), ("Vận chuyển 2", "Vận chuyển"),
            ("Nhà hàng A", "Ăn uống"), ("Nhà hàng B", "Ăn uống"),
        ]

        # route -> possible units (simple mapping)
        self.route_unit_pairs = [(r, random.choice(self.business_units)) for r in self.tour_routes]

    def generate_tour_data(self, start_date, end_date, num_tours=1500):
        tours = []
        num_customers = int(num_tours * 0.7)
        customer_ids = [f"KH{i:06d}" for i in range(1, num_customers + 1)]

        for i in range(num_tours):
            booking_date = fake.date_time_between(start_date=start_date, end_date=end_date).replace(tzinfo=None)
            route, business_unit = random.choice(self.route_unit_pairs)
            channel = random.choices(self.sales_channels, weights=[0.35, 0.40, 0.25])[0]
            segment = random.choices(self.segments, weights=[0.35, 0.55, 0.10])[0]

            if random.random() < 0.3:
                num_customers_in_booking = random.randint(2, 4)
            else:
                num_customers_in_booking = random.randint(5, 20)

            tour_capacity = random.choice([20, 25, 30, 35, 40, 45])
            price_per_person = random.randint(3000000, 15000000)
            revenue = price_per_person * num_customers_in_booking
            cost_ratio = random.uniform(0.85, 0.95)
            cost = revenue * cost_ratio
            gross_profit = revenue - cost
            gross_profit_margin = (gross_profit / revenue * 100) if revenue > 0 else 0

            status = random.choices(["Đã xác nhận", "Đã hủy", "Hoãn"], weights=[0.75, 0.15, 0.10])[0]
            customer_id = random.choice(customer_ids)

            if channel == "Online":
                marketing_cost = revenue * random.uniform(0.02, 0.05)
                sales_cost = revenue * random.uniform(0.01, 0.02)
            else:
                marketing_cost = revenue * random.uniform(0.01, 0.03)
                sales_cost = revenue * random.uniform(0.02, 0.05)

            opex = marketing_cost + sales_cost
            partner_name, partner_type = random.choice(self.partners)

            tours.append({
                'booking_id': f"BK{i+1:06d}",
                'customer_id': customer_id,
                'booking_date': booking_date,
                'route': route,
                'business_unit': business_unit,
                'sales_channel': channel,
                'segment': segment,
                'num_customers': num_customers_in_booking,
                'tour_capacity': tour_capacity,
                'price_per_person': price_per_person,
                'revenue': revenue,
                'cost': cost,
                'gross_profit': gross_profit,
                'gross_profit_margin': gross_profit_margin,
                'status': status,
                'marketing_cost': marketing_cost,
                'sales_cost': sales_cost,
                'opex': opex,
                'partner': partner_name,
                'partner_type': partner_type,
                'service_type': partner_type,
                'contract_status': random.choice(["Đang triển khai", "Sắp hết hạn", "Đã thanh lý"]),
                'payment_status': random.choice(["Trả trước", "Trả sau", "Chưa thanh toán"]),
                'feedback_ratio': random.uniform(0.7, 0.95),
                'customer_age_group': random.choice(["18-25 (Gen Z)", "26-35 (Young Pro)", "36-55 (Mid-Career)", "56+ (Retiree)"]),
                'customer_nationality': random.choice(["Việt Nam", "Hàn Quốc", "Trung Quốc"]),
                'service_cost': cost * random.uniform(0.8, 1.2)
            })

        return pd.DataFrame(tours)

    def generate_plan_data(self, year, month=None):
        plans = []
        if month:
            periods = [(year, month)]
        else:
            periods = [(year, m) for m in range(1, 13)]

        for y, m in periods:
            for bu in self.business_units:
                for route in self.tour_routes:
                    for seg in self.segments:
                        seasonality = random.uniform(0.8, 1.2)
                        base_customers = random.randint(5, 20)
                        planned_customers = int(base_customers * seasonality)
                        avg_price = random.randint(3000000, 15000000)
                        planned_revenue = planned_customers * avg_price
                        plans.append({
                            'year': y,
                            'month': m,
                            'business_unit': bu,
                            'route': route,
                            'segment': seg,
                            'planned_customers': planned_customers,
                            'planned_revenue': planned_revenue,
                            'planned_gross_profit': planned_revenue * 0.2
                        })
        return pd.DataFrame(plans)

    def generate_historical_data(self, current_date, lookback_years=2):
        all_data = []
        for year_offset in range(lookback_years + 1):
            year_start = datetime(current_date.year - year_offset, 1, 1)
            year_end = datetime(current_date.year - year_offset, 12, 31)
            num_tours = random.randint(400, 600)
            all_data.append(self.generate_tour_data(year_start, year_end, num_tours=num_tours))
        return pd.concat(all_data, ignore_index=True)


# -- Helpers for Google Sheet parsing --
def _col_index(letter):
    """Convert Excel column letter to 0-based index (A->0)."""
    s = 0
    for ch in letter.upper():
        s = s * 26 + (ord(ch) - ord('A') + 1)
    return s - 1


def _parse_number(val):
    try:
        if pd.isna(val):
            return None
        if isinstance(val, str):
            v = val.strip().replace(',', '').replace(' ', '')
            v = v.replace('₫', '').replace('$', '').replace('(', '-').replace(')', '')
            if v == '':
                return None
            return float(v)
        return float(val)
    except Exception:
        return None


def _get_spreadsheet_id(url):
    try:
        return url.split('/d/')[1].split('/')[0]
    except Exception:
        return None


def load_or_generate_data(spreadsheet_url=None):
    """Load data from Google Sheet (public) or generate mock data.

    Returns: (tours_df, plans_df, historical_df, meta)
    meta contains keys: used_sheet (bool), processed_files (list), parsed_rows (int), parsed_counts (dict)
    """
    generator = VietravelDataGenerator()
    current_date = datetime.now()
    current_year = current_date.year

    tours_records = []
    parsed_counts = {}
    processed_files = []

    if spreadsheet_url:
        sheet_id = _get_spreadsheet_id(spreadsheet_url)
        if sheet_id:
            # Try to parse gid if present in URL
            gid = None
            if 'gid=' in spreadsheet_url:
                try:
                    # Handle cases like ...edit?gid=123#gid=123 or additional params
                    gid_part = spreadsheet_url.split('gid=')[1]
                    gid = gid_part.split('&')[0].split('#')[0]
                except Exception:
                    gid = None
            gid = gid or '0'
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

            try:
                resp = requests.get(csv_url, timeout=15)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.content.decode('utf-8')))
                processed_files.append(csv_url)

                # Column positions (0-based) from Excel letters provided by user
                # E=4, F=5, G=6, I=8, J=9, P=15, Q=16, R=17, S=18
                idx_map = {
                    'E': _col_index('E'),
                    'F': _col_index('F'),
                    'G': _col_index('G'),
                    'I': _col_index('I'),
                    'J': _col_index('J'),
                    'P': _col_index('P'),
                    'Q': _col_index('Q'),
                    'R': _col_index('R'),
                    'S': _col_index('S'),
                }

                per_file_count = 0
                # Iterate using iloc to access by positional column index
                for _, row in df.iterrows():
                    try:
                        # use iloc to be robust against header names
                        start_date = pd.to_datetime(row.iloc[idx_map['E']]) if pd.notna(row.iloc[idx_map['E']]) else None
                        end_date = pd.to_datetime(row.iloc[idx_map['F']]) if pd.notna(row.iloc[idx_map['F']]) else None
                        num_booked = int(_parse_number(row.iloc[idx_map['G']]) or 0)
                        revenue = float(_parse_number(row.iloc[idx_map['I']]) or 0)
                        gross_profit = float(_parse_number(row.iloc[idx_map['J']]) or 0)
                        route = str(row.iloc[idx_map['P']]).strip() if pd.notna(row.iloc[idx_map['P']]) else None
                        business_unit = str(row.iloc[idx_map['Q']]).strip() if pd.notna(row.iloc[idx_map['Q']]) else None
                        tour_capacity = int(_parse_number(row.iloc[idx_map['R']]) or 0)
                        segment = str(row.iloc[idx_map['S']]).strip() if pd.notna(row.iloc[idx_map['S']]) else None

                        if not route or not business_unit:
                            # skip incomplete rows
                            continue

                        per_file_count += 1
                        booking_id = f"GS{per_file_count:06d}"
                        customer_id = f"KHG{per_file_count:06d}"
                        booking_date = start_date if start_date is not None else datetime.now()

                        price_per_person = revenue / num_booked if num_booked > 0 else 0
                        cost = revenue - gross_profit if (revenue and gross_profit) else revenue * 0.8
                        gross_profit_margin = (gross_profit / revenue * 100) if revenue > 0 else 0

                        sales_channel = random.choice(generator.sales_channels)
                        status = "Đã xác nhận"

                        if sales_channel == "Online":
                            marketing_cost = revenue * random.uniform(0.02, 0.05)
                            sales_cost = revenue * random.uniform(0.01, 0.02)
                        else:
                            marketing_cost = revenue * random.uniform(0.01, 0.03)
                            sales_cost = revenue * random.uniform(0.02, 0.04)

                        opex = marketing_cost + sales_cost

                        partner_name, partner_type = random.choice(generator.partners)
                        service_type = partner_type
                        contract_status = random.choices(["Đang triển khai", "Sắp hết hạn", "Đã thanh lý"], weights=[0.8, 0.1, 0.1])[0]
                        payment_status = random.choices(["Trả trước", "Trả sau", "Chưa thanh toán"], weights=[0.6, 0.3, 0.1])[0]
                        feedback_ratio = random.uniform(0.7, 0.95)
                        service_cost = cost * random.uniform(0.8, 1.2)

                        tours_records.append({
                            'booking_id': booking_id,
                            'customer_id': customer_id,
                            'booking_date': booking_date,
                            'route': route,
                            'business_unit': business_unit,
                            'sales_channel': sales_channel,
                            'segment': segment,
                            'num_customers': int(num_booked),
                            'tour_capacity': int(tour_capacity),
                            'price_per_person': float(price_per_person),
                            'revenue': float(revenue),
                            'cost': float(cost),
                            'gross_profit': float(gross_profit),
                            'gross_profit_margin': float(gross_profit_margin),
                            'status': status,
                            'marketing_cost': float(marketing_cost),
                            'sales_cost': float(sales_cost),
                            'opex': float(opex),
                            'partner': partner_name,
                            'partner_type': partner_type,
                            'service_type': service_type,
                            'contract_status': contract_status,
                            'payment_status': payment_status,
                            'feedback_ratio': feedback_ratio,
                            'customer_age_group': random.choice(["18-25 (Gen Z)", "26-35 (Young Pro)", "36-55 (Mid-Career)", "56+ (Retiree)"]),
                            'customer_nationality': random.choice(["Việt Nam", "Hàn Quốc", "Trung Quốc"]),
                            'service_cost': float(service_cost)
                        })

                    except Exception:
                        # Skip problematic row but continue
                        continue

                parsed_counts[csv_url] = per_file_count

            except Exception:
                # If fetching/reading sheet failed, fall back to generator
                parsed_counts = {}

    # Build DataFrames
    if tours_records:
        tours_df = pd.DataFrame(tours_records)
        used_sheet = True
    else:
        year_start = datetime(current_year, 1, 1)
        year_end = current_date
        tours_df = generator.generate_tour_data(year_start, year_end, num_tours=1500)
        used_sheet = False

    plans_df = generator.generate_plan_data(current_year)
    historical_df = generator.generate_historical_data(current_date, lookback_years=2)

    meta = {
        'used_sheet': used_sheet,
        'processed_files': processed_files,
        'parsed_rows': len(tours_records),
        'parsed_counts': parsed_counts
    }

    return tours_df, plans_df, historical_df, meta
