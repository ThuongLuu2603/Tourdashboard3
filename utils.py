"""
Utility functions for Vietravel Business Intelligence Dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def format_currency(value):
    """Format number as Vietnamese currency (VND)"""
    if pd.isna(value) or value is None:
        return "0 ₫"
    
    # Convert to billions for readability if > 1 billion
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.1f} tỷ ₫"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.1f} triệu ₫"
    else:
        return f"{value:,.0f} ₫"


def format_number(value):
    """Format number with thousand separators"""
    if pd.isna(value) or value is None:
        return "0"
    return f"{int(value):,}"


def format_percentage(value):
    """Format number as percentage"""
    if pd.isna(value) or value is None:
        return "0%"
    # Sửa: Bảo vệ giá trị âm hoặc NaN
    return f"{max(0, value):.1f}%" 


def calculate_completion_rate(actual, planned):
    """Calculate completion rate percentage"""
    if planned == 0 or pd.isna(planned) or planned is None:
        return 0
    return (actual / planned) * 100


def get_growth_rate(current, previous):
    """Calculate growth rate percentage"""
    if previous == 0 or pd.isna(previous) or previous is None:
        return 0
    return ((current - previous) / previous) * 100


def filter_data_by_date(df, start_date, end_date, date_column='booking_date'):
    """Filter dataframe by date range"""
    
    if df.empty or date_column not in df.columns:
        return pd.DataFrame(columns=df.columns)
        
    mask = (df[date_column] >= pd.to_datetime(start_date)) & \
           (df[date_column] <= pd.to_datetime(end_date))
    return df[mask].copy()


def filter_confirmed_bookings(df):
    """Filter only confirmed bookings (exclude cancelled/postponed)"""
    if 'status' not in df.columns:
        return pd.DataFrame(columns=df.columns)
    return df[df['status'] == 'Đã xác nhận'].copy()


def calculate_kpis(tours_df, plans_df, start_date, end_date):
    """
    Calculate key performance indicators for the dashboard
    """
    # Filter data for current period
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    
    # Calculate actual metrics
    actual_revenue = confirmed_data['revenue'].sum()
    actual_gross_profit = confirmed_data['gross_profit'].sum()
    actual_customers = confirmed_data['num_customers'].sum()
    
    # Filter plans for the same period
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    plan_mask = (plans_df['year'] == start_dt.year) & \
                (plans_df['month'] >= start_dt.month) & \
                (plans_df['month'] <= end_dt.month)
    period_plans = plans_df[plan_mask]
    
    planned_revenue = period_plans['planned_revenue'].sum()
    planned_gross_profit = period_plans['planned_gross_profit'].sum()
    planned_customers = period_plans['planned_customers'].sum()
    
    # Calculate same period last year
    last_year_start = start_dt - timedelta(days=365)
    last_year_end = end_dt - timedelta(days=365)
    last_year_data = filter_data_by_date(tours_df, last_year_start, last_year_end)
    last_year_confirmed = filter_confirmed_bookings(last_year_data)
    
    ly_revenue = last_year_confirmed['revenue'].sum()
    ly_gross_profit = last_year_confirmed['gross_profit'].sum()
    ly_customers = last_year_confirmed['num_customers'].sum()
    
    # Completion rates
    revenue_completion = calculate_completion_rate(actual_revenue, planned_revenue)
    customer_completion = calculate_completion_rate(actual_customers, planned_customers)
    
    # Growth rates
    revenue_growth = get_growth_rate(actual_revenue, ly_revenue)
    profit_growth = get_growth_rate(actual_gross_profit, ly_gross_profit)
    customer_growth = get_growth_rate(actual_customers, ly_customers)
    
    return {
        'actual_revenue': actual_revenue,
        'actual_gross_profit': actual_gross_profit,
        'actual_customers': actual_customers,
        'planned_revenue': planned_revenue,
        'planned_gross_profit': planned_gross_profit,
        'planned_customers': planned_customers,
        'ly_revenue': ly_revenue,
        'ly_gross_profit': ly_gross_profit,
        'ly_customers': ly_customers,
        'revenue_completion': revenue_completion,
        'customer_completion': customer_completion,
        'revenue_growth': revenue_growth,
        'profit_growth': profit_growth,
        'customer_growth': customer_growth
    }


def create_gauge_chart(value, title, max_value=150, threshold=100, unit_breakdown=None, is_inverse_metric=False, actual=None, planned=None):
    """Create a gauge chart for completion rate with hover info for business units and actual/planned display"""
    
    value = value if not pd.isna(value) else 0

    # LÔ-GÍC MÀU ĐÃ ĐẢO NGƯỢC
    if is_inverse_metric:
        if value <= threshold:
            color = "#00CC96"  # Xanh lá: Tỷ lệ Tốt (Dưới ngưỡng)
            bgcolor = "rgba(0, 204, 150, 0.2)"
        elif value <= threshold * 1.5:
            color = "#FFA500"  # Cam: Cần chú ý
            bgcolor = "rgba(255, 165, 0, 0.2)"
        else:
            color = "#EF553B"  # Đỏ: Xấu (Vượt xa ngưỡng)
            bgcolor = "rgba(239, 85, 59, 0.2)"
    else:
        # Logic màu ban đầu (Doanh thu, Lượt khách)
        if value >= threshold:
            color = "#00CC96"
            bgcolor = "rgba(0, 204, 150, 0.2)"
        elif value >= threshold * 0.8:
            color = "#FFA500"
            bgcolor = "rgba(255, 165, 0, 0.2)"
        else:
            color = "#EF553B"
            bgcolor = "rgba(239, 85, 59, 0.2)"
    
    # Build title with actual/planned if provided
    title_text = title
    if actual is not None and planned is not None:
        # Format numbers based on magnitude
        if actual >= 1_000_000_000 or planned >= 1_000_000_000:  # Billions
            actual_fmt = f"{actual/1_000_000_000:.1f}B"
            planned_fmt = f"{planned/1_000_000_000:.1f}B"
        elif actual >= 1_000_000 or planned >= 1_000_000:  # Millions
            actual_fmt = f"{actual/1_000_000:.1f}M"
            planned_fmt = f"{planned/1_000_000:.1f}M"
        elif actual >= 1_000 or planned >= 1_000:  # Thousands
            actual_fmt = f"{actual/1_000:.1f}K"
            planned_fmt = f"{planned/1_000:.1f}K"
        else:
            actual_fmt = f"{actual:.0f}"
            planned_fmt = f"{planned:.0f}"
        
        title_text = f"{title}<br><span style='font-size:10px; color:#666;'>{actual_fmt} / {planned_fmt}</span>"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [1, 1]},
        title = {'text': title_text, 'font': {'size': 11}},
        number = {
            'suffix': "%", 
            'font': {'size': 20}
        },
        gauge = {
            'axis': {'range': [None, max_value], 'ticksuffix': "%", 'tickfont': {'size': 9}},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold * 0.5], 'color': "#FFE5E5"},
                {'range': [threshold * 0.5, threshold * 0.8], 'color': "#FFF4E5"},
                {'range': [threshold * 0.8, threshold], 'color': "#E5F5E5"},
                {'range': [threshold, max_value], 'color': "#D4F1D4"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    
    # Add invisible scatter trace for hover info with business unit breakdown
    if unit_breakdown is not None and not unit_breakdown.empty:
        hover_text = "Chi tiết theo đơn vị:<br>"
        for _, row in unit_breakdown.iterrows():
            hover_text += f"<br>{row['business_unit']}: {row['completion']:.1f}%"
        
        fig.add_trace(go.Scatter(
            x=[0.5],
            y=[0.1],
            mode='markers',
            marker=dict(size=100, color='rgba(0,0,0,0)', opacity=0),
            hovertemplate=hover_text + "<extra></extra>",
            showlegend=False
        ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=5),
        hovermode='closest'
    )
    
    return fig


def create_bar_chart(data, x, y, title, orientation='v', color=None):
    """Create a bar chart"""
    
    if orientation == 'h':
        fig = px.bar(data, x=y, y=x, orientation='h', title=title, color=color,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="")
    else:
        fig = px.bar(data, x=x, y=y, title=title, color=color,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="")
    
    fig.update_layout(
        height=400,
        showlegend=True if color else False,
        hovermode='x unified'
    )
    
    return fig


def create_pie_chart(data, values, names, title):
    """Create a pie chart"""
    
    fig = px.pie(data, values=values, names=names, title=title,
                 color_discrete_sequence=px.colors.qualitative.Set2)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        height=400,
        showlegend=True
    )
    
    return fig


def create_line_chart(data, x, y, title, color=None):
    """Create a line chart"""
    
    fig = px.line(data, x=x, y=y, title=title, color=color,
                  markers=True, color_discrete_sequence=px.colors.qualitative.Set2)
    
    fig.update_xaxes(title="")
    fig.update_yaxes(title="")
    fig.update_layout(
        height=400,
        showlegend=True if color else False,
        hovermode='x unified'
    )
    
    return fig


def get_top_routes(tours_df, n=10, metric='revenue'):
    """
    Get top N routes by specified metric
    """
    confirmed = filter_confirmed_bookings(tours_df)
    
    if confirmed.empty: 
        return pd.DataFrame(columns=['route', 'revenue', 'num_customers', 'gross_profit', 'profit_margin'])

    if metric == 'revenue':
        grouped = confirmed.groupby('route').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        grouped = grouped.sort_values('revenue', ascending=False).head(n)
    elif metric == 'customers':
        grouped = confirmed.groupby('route').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        grouped = grouped.sort_values('num_customers', ascending=False).head(n)
    else:  # gross_profit
        grouped = confirmed.groupby('route').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        grouped = grouped.sort_values('gross_profit', ascending=False).head(n)
    
    # ĐÃ SỬA: Bảo vệ chia cho 0
    grouped['profit_margin'] = np.where(
        grouped['revenue'] > 0,
        (grouped['gross_profit'] / grouped['revenue'] * 100).round(2),
        0
    )
    
    return grouped


def get_route_unit_breakdown(tours_df, route_name, metric='revenue'):
    """
    Get breakdown by business unit for a specific route and metric
    """
    confirmed = filter_confirmed_bookings(tours_df)
    route_data = confirmed[confirmed['route'] == route_name]
    
    if route_data.empty:
        return pd.DataFrame(columns=['business_unit', 'revenue', 'num_customers', 'gross_profit', 'percentage'])
    
    unit_breakdown = route_data.groupby('business_unit').agg({
        'revenue': 'sum',
        'num_customers': 'sum',
        'gross_profit': 'sum'
    }).reset_index()
    
    if metric == 'revenue':
        total_value = unit_breakdown['revenue'].sum()
        col_name = 'revenue'
    elif metric == 'customers':
        total_value = unit_breakdown['num_customers'].sum()
        col_name = 'num_customers'
    else: # profit
        total_value = unit_breakdown['gross_profit'].sum()
        col_name = 'gross_profit'
    
    # SỬA LỖI LOGIC: Phải dùng col_name để tính tỷ trọng, không phải luôn là 'revenue'
    unit_breakdown['percentage'] = np.where(
        total_value > 0,
        (unit_breakdown[col_name] / total_value * 100).round(1), # ĐÃ SỬA: Dùng col_name
        0
    )
    unit_breakdown = unit_breakdown.sort_values(col_name, ascending=False)
    
    return unit_breakdown


def calculate_operational_metrics(tours_df):
    """
    Calculate operational metrics
    """
    # Average occupancy rate
    confirmed = filter_confirmed_bookings(tours_df)
    total_booked = confirmed['num_customers'].sum()
    total_capacity = confirmed['tour_capacity'].sum()
    # Hàm này đã có bảo vệ chia cho 0
    avg_occupancy = (total_booked / total_capacity * 100) if total_capacity > 0 else 0
    
    # Cancellation/postponement rate
    total_bookings = len(tours_df)
    cancelled_postponed = len(tours_df[tours_df['status'].isin(['Đã hủy', 'Hoãn'])])
    # Hàm này đã có bảo vệ chia cho 0
    cancel_rate = (cancelled_postponed / total_bookings * 100) if total_bookings > 0 else 0
    
    # Returning customer rate
    customer_counts = tours_df.groupby('customer_id').size()
    returning_customers = len(customer_counts[customer_counts >= 2])
    total_unique_customers = len(customer_counts)
    # Hàm này đã có bảo vệ chia cho 0
    returning_rate = (returning_customers / total_unique_customers * 100) if total_unique_customers > 0 else 0
    
    return {
        'avg_occupancy': avg_occupancy,
        'cancel_rate': cancel_rate,
        'returning_rate': returning_rate
    }


def get_low_margin_tours(tours_df, threshold=5):
    """
    Get tours with profit margin below threshold
    """
    confirmed = filter_confirmed_bookings(tours_df)
    
    if confirmed.empty:
        return pd.DataFrame(columns=['route', 'gross_profit', 'revenue', 'num_customers', 'profit_margin'])

    # Group by route and calculate average margin
    route_margins = confirmed.groupby('route').agg({
        'gross_profit': 'sum',
        'revenue': 'sum',
        'num_customers': 'sum'
    }).reset_index()
    
    # ĐÃ SỬA: Bảo vệ chia cho 0
    route_margins['profit_margin'] = np.where(
        route_margins['revenue'] > 0,
        (route_margins['gross_profit'] / route_margins['revenue'] * 100),
        0
    )
    
    # Filter low margin routes
    low_margin = route_margins[route_margins['profit_margin'] < threshold].sort_values('profit_margin')
    
    return low_margin


def get_unit_performance(tours_df, plans_df, start_date, end_date):
    """
    Calculate performance by business unit
    """
    # Filter current period data
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    
    if confirmed_data.empty: 
        return pd.DataFrame(columns=['business_unit', 'actual_revenue', 'actual_profit', 'actual_customers', 'planned_revenue', 'planned_gross_profit', 'planned_customers', 'revenue_completion', 'customer_completion', 'profit_margin'])

    # Actual by unit
    actual_by_unit = confirmed_data.groupby('business_unit').agg({
        'revenue': 'sum',
        'gross_profit': 'sum',
        'num_customers': 'sum'
    }).reset_index()
    actual_by_unit.columns = ['business_unit', 'actual_revenue', 'actual_profit', 'actual_customers']
    
    # Plans by unit
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    plan_mask = (plans_df['year'] == start_dt.year) & \
                (plans_df['month'] >= start_dt.month) & \
                (plans_df['month'] <= end_dt.month)
    period_plans = plans_df[plan_mask]
    
    plan_by_unit = period_plans.groupby('business_unit').agg({
        'planned_revenue': 'sum',
        'planned_gross_profit': 'sum',
        'planned_customers': 'sum'
    }).reset_index()
    
    # Merge and calculate completion
    performance = actual_by_unit.merge(plan_by_unit, on='business_unit', how='left').fillna(0)
    
    # ĐÃ SỬA: Bảo vệ chia cho 0
    performance['revenue_completion'] = np.where(
        performance['planned_revenue'] > 0,
        (performance['actual_revenue'] / performance['planned_revenue'] * 100),
        0
    )
    performance['customer_completion'] = np.where(
        performance['planned_customers'] > 0,
        (performance['actual_customers'] / performance['planned_customers'] * 100),
        0
    )
    performance['profit_margin'] = np.where(
        performance['actual_revenue'] > 0,
        (performance['actual_profit'] / performance['actual_revenue'] * 100),
        0
    )
    
    return performance


def get_unit_breakdown(tours_df, plans_df, start_date, end_date, metric='revenue'):
    """
    Get completion rate breakdown by business unit for a specific metric (Dùng cho Gauge Chart Hover)
    """
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    plan_mask = (plans_df['year'] == start_dt.year) & \
                (plans_df['month'] >= start_dt.month) & \
                (plans_df['month'] <= end_dt.month)
    period_plans = plans_df[plan_mask]
    
    results = []
    for unit in sorted(confirmed_data['business_unit'].unique()):
        unit_data = confirmed_data[confirmed_data['business_unit'] == unit]
        unit_plans = period_plans[period_plans['business_unit'] == unit]
        
        if metric == 'revenue':
            actual = unit_data['revenue'].sum()
            planned = unit_plans['planned_revenue'].sum()
        elif metric == 'profit':
            actual = unit_data['gross_profit'].sum()
            planned = unit_plans['planned_gross_profit'].sum()
        else:  # customers
            actual = unit_data['num_customers'].sum()
            planned = unit_plans['planned_customers'].sum()
        
        completion = calculate_completion_rate(actual, planned)
        results.append({
            'business_unit': unit,
            'completion': completion
        })
    
    return pd.DataFrame(results)


def get_segment_breakdown(tours_df, start_date, end_date, metric='revenue'):
    """
    Get breakdown by segment (FIT/GIT/Inbound) for a specific metric
    """
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    
    if confirmed_data.empty:
        return pd.DataFrame(columns=['segment', 'value', 'percentage'])
    
    if metric == 'revenue':
        segment_data = confirmed_data.groupby('segment')['revenue'].sum().reset_index()
        segment_data.columns = ['segment', 'value']
    elif metric == 'customers':
        segment_data = confirmed_data.groupby('segment')['num_customers'].sum().reset_index()
        segment_data.columns = ['segment', 'value']
    else:  # profit
        segment_data = confirmed_data.groupby('segment')['gross_profit'].sum().reset_index()
        segment_data.columns = ['segment', 'value']
    
    total_value = segment_data['value'].sum()
    # ĐÃ SỬA: Bảo vệ chia cho 0
    segment_data['percentage'] = np.where(
        total_value > 0,
        (segment_data['value'] / total_value * 100).round(1),
        0
    )
    segment_data = segment_data.sort_values('value', ascending=False)
    
    return segment_data


def get_segment_unit_breakdown(tours_df, start_date, end_date, segment_name, metric='revenue'):
    """
    Get business unit breakdown for a specific segment (for hover tooltips)
    """
    current_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(current_data)
    segment_data = confirmed_data[confirmed_data['segment'] == segment_name]
    
    if segment_data.empty:
        return pd.DataFrame(columns=['business_unit', 'value', 'percentage'])
    
    if metric == 'revenue':
        unit_breakdown = segment_data.groupby('business_unit')['revenue'].sum().reset_index()
        unit_breakdown.columns = ['business_unit', 'value']
    elif metric == 'customers':
        unit_breakdown = segment_data.groupby('business_unit')['num_customers'].sum().reset_index()
        unit_breakdown.columns = ['business_unit', 'value']
    else:  # profit
        unit_breakdown = segment_data.groupby('business_unit')['gross_profit'].sum().reset_index()
        unit_breakdown.columns = ['business_unit', 'value']
    
    total_value = unit_breakdown['value'].sum()
    # ĐÃ SỬA: Bảo vệ chia cho 0
    unit_breakdown['percentage'] = np.where(
        total_value > 0,
        (unit_breakdown['value'] / total_value * 100).round(1),
        0
    )
    unit_breakdown = unit_breakdown.sort_values('value', ascending=False)
    
    return unit_breakdown


def create_forecast_chart(tours_df, plans_df, start_date, end_date, date_option):
    """
    Create forecast chart combining cumulative actuals (bars) and planned/forecast lines (lines) for revenue.
    Requires date_option (Tuần/Tháng/Quý/Năm) to determine the period_end_dt.
    """
    
    # --- 1. Chuẩn bị Dữ liệu và Chuẩn hóa Ngày tháng ---
    confirmed_data = filter_confirmed_bookings(tours_df)
    
    start_dt = pd.to_datetime(start_date).normalize()
    end_dt = pd.to_datetime(end_date).normalize()
    today = pd.to_datetime(datetime.now().date())
    
    # LÔ-GÍC XÁC ĐỊNH NGÀY CUỐI CÙNG VÀ ĐỘ PHÂN GIẢI
    period_end_dt = today
    
    # Xác định độ phân giải (Granularity)
    if date_option == 'Năm' or date_option == 'Quý':
        freq_unit = 'M'
        x_format = "%m/%Y"
        x_title = "Tháng"
        
        # Mở rộng kỳ Dự báo
        if date_option == 'Năm':
            period_end_dt = pd.to_datetime(datetime(start_dt.year, 12, 31))
        elif date_option == 'Quý':
            # Dự báo đến ngày cuối cùng của quý (Quý IV bắt đầu 01/10)
            if start_dt.month in [1, 4, 7, 10]:
                end_month = start_dt.month + 2
                period_end_dt = pd.to_datetime(datetime(start_dt.year, end_month, 1)) + pd.offsets.MonthEnd(0)
            else:
                period_end_dt = today
        
    elif date_option == 'Tháng':
        freq_unit = 'W' 
        x_format = "T%W"
        x_title = "Tuần"
        
        # Dự báo đến ngày cuối cùng của tháng
        period_end_dt = start_dt + pd.offsets.MonthEnd(0)
        
    elif date_option == 'Tuần' or date_option == 'Tùy chỉnh':
        freq_unit = 'D'
        x_format = "%d/%m"
        x_title = "Ngày"
        period_end_dt = end_dt
        
    else: 
        freq_unit = 'D'
        x_format = "%d/%m"
        x_title = "Ngày"
        period_end_dt = end_dt


    # Lọc dữ liệu Thực hiện đến ngày hôm nay
    period_tours = filter_data_by_date(confirmed_data, start_dt, today, date_column='booking_date')

    if period_tours.empty:
        return go.Figure().update_layout(title=f'Không có dữ liệu Thực hiện từ {start_dt.strftime("%d/%m")}', height=300)
        
    # --- 2. Xử lý Dữ liệu Lũy kế Thực hiện ---
    
    # Tổng hợp Actuals theo đơn vị thời gian (freq_unit)
    period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period(freq_unit)
    daily_actual = period_tours.groupby('period')[['revenue']].sum().reset_index()
    
    # Chuyển Period sang Timestamp để vẽ cột
    if freq_unit == 'M':
        # Dùng ngày đầu tháng để vẽ cột
        daily_actual['date'] = daily_actual['period'].apply(lambda x: x.start_time.normalize())
        # Chiều rộng 20 ngày cho cột tháng
        bar_width = 20 * 24 * 60 * 60 * 1000 
    elif freq_unit == 'W':
        daily_actual['date'] = daily_actual['period'].apply(lambda x: x.start_time.normalize())
        # Chiều rộng 5 ngày cho cột tuần
        bar_width = 5 * 24 * 60 * 60 * 1000
    else: # D (Ngày)
        daily_actual['date'] = daily_actual['period'].apply(lambda x: x.end_time.normalize())
        bar_width = None # Plotly tự quyết định cho ngày
        
    # Sắp xếp và tính lũy kế
    daily_actual = daily_actual.sort_values('date')
    daily_actual['cumulative_actual'] = daily_actual['revenue'].cumsum()
    
    actual_data_points = daily_actual.copy()
    
    # Giá trị tổng thực hiện chính xác đến hôm nay (dùng cho Run-rate)
    current_actual_revenue = period_tours['revenue'].sum() 


    # --- 3. Xử lý Dữ liệu Kế hoạch và Dự báo ---
    
    plan_mask = (plans_df['year'] == start_dt.year) & \
                (plans_df['month'] >= start_dt.month) & \
                (plans_df['month'] <= period_end_dt.month) 
                
    total_planned_revenue = plans_df[plan_mask]['planned_revenue'].sum()

    # Tính Kế hoạch lũy kế tuyến tính (Planned Line)
    plan_date_range = pd.date_range(start=start_dt, end=period_end_dt, freq='D') 
    total_days_in_period = (period_end_dt - start_dt).days + 1
    daily_plan_rate = total_planned_revenue / total_days_in_period if total_days_in_period > 0 else 0
    
    daily_plan_line = pd.DataFrame({'date': plan_date_range})
    daily_plan_line['cumulative_planned'] = (daily_plan_line['date'] - start_dt).dt.days * daily_plan_rate + daily_plan_rate
    
    # Tính Dự báo Run-rate 
    days_elapsed = (today - start_dt).days + 1
    daily_run_rate = current_actual_revenue / days_elapsed if days_elapsed > 0 else 0
    
    # Tạo chuỗi Dự báo (Forecast Line)
    forecast_dates = pd.date_range(start=today, end=period_end_dt, freq='D')
    forecast_line = pd.DataFrame({'date': forecast_dates})

    forecast_line['cumulative_forecast'] = current_actual_revenue + (
        (forecast_line['date'] - today).dt.days * daily_run_rate
    )
    
    # --- 4. Tạo Biểu đồ Kết hợp ---
    
    fig = go.Figure()

    # Trace 1: Thực hiện Lũy kế (Dạng cột - ĐÃ SỬA LỖI WIDTH)
    fig.add_trace(go.Bar(
        x=actual_data_points['date'],
        y=actual_data_points['cumulative_actual'],
        name='Thực hiện Lũy kế',
        marker_color='#636EFA',
        width=bar_width,
        hovertemplate=f'{x_title}: %{{x|{x_format}}}<br>Thực hiện: %{{y:,.0f}} ₫<extra></extra>'
    ))
    
    # Trace 2: Kế hoạch Lũy kế (Đường)
    fig.add_trace(go.Scatter(
        x=daily_plan_line['date'],
        y=daily_plan_line['cumulative_planned'],
        name='Kế hoạch Lũy kế',
        mode='lines',
        line=dict(color='#EF553B', width=2),
        hovertemplate='Ngày: %{x|%d/%m}<br>Kế hoạch: %{y:,.0f} ₫<extra></extra>'
    ))

    # Trace 3: Đường Dự báo Cuối kỳ (Đường nét đứt)
    anchor_point = pd.DataFrame({
        'date': [today],
        'cumulative_forecast': [current_actual_revenue]
    })
    
    forecast_dates_extended = pd.concat([anchor_point, 
                                         forecast_line[['date', 'cumulative_forecast']]], ignore_index=True)
                                         
    fig.add_trace(go.Scatter(
        x=forecast_dates_extended['date'],
        y=forecast_dates_extended['cumulative_forecast'],
        name='Dự báo Cuối kỳ',
        mode='lines',
        line=dict(color='#00CC96', width=2, dash='dot'),
        hovertemplate='Ngày: %{x|%d/%m}<br>Dự báo: %{y:,.0f} ₫<extra></extra>'
    ))
    
    # --- 5. Cập nhật Layout và Định dạng ---
    
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Doanh thu Lũy kế (₫)",
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=40, b=30),
        hovermode='x unified'
    )
    
    fig.update_xaxes(tickformat=x_format, title=x_title, tickangle=0)
    
    return fig

def create_trend_chart(tours_df, start_date, end_date, metrics=['revenue', 'customers', 'profit']):
    """
    Create a multi-line trend chart showing trends over time
    """
    # Filter confirmed bookings in period
    confirmed_tours = filter_confirmed_bookings(tours_df)
    period_tours = filter_data_by_date(confirmed_tours, start_date, end_date)
    
    # Calculate period length in days
    period_length = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    
    # Determine grouping granularity
    if period_length <= 7:
        # Daily granularity for week or less
        period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('D')
        period_data = period_tours.groupby('period').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        period_data['period_str'] = period_data['period'].dt.strftime('%d/%m')
        x_title = "Ngày"
    elif period_length <= 60:
        # Weekly granularity for 2 months or less
        period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('W')
        period_data = period_tours.groupby('period').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        period_data['period_str'] = period_data['period'].apply(lambda x: f"T{x.week}")
        x_title = "Tuần"
    else:
        # Monthly granularity for longer periods
        period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('M')
        period_data = period_tours.groupby('period').agg({
            'revenue': 'sum',
            'num_customers': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        period_data['period_str'] = period_data['period'].astype(str)
        x_title = "Tháng"
    
    monthly_data = period_data
    
    # Create figure
    fig = go.Figure()
    
    # Helper function to format values
    def format_value(val):
        if val >= 1e9:
            return f'{val/1e9:.1f}B'
        elif val >= 1e6:
            return f'{val/1e6:.1f}M'
        elif val >= 1e3:
            return f'{val/1e3:.0f}K'
        else:
            return f'{val:.0f}'
    
    if 'revenue' in metrics:
        fig.add_trace(go.Scatter(
            x=monthly_data['period_str'],
            y=monthly_data['revenue'],
            name='Doanh thu',
            mode='lines+markers+text',
            line=dict(color='#636EFA', width=2),
            text=monthly_data['revenue'].apply(format_value),
            textposition='top center',
            yaxis='y1'
        ))
    
    if 'customers' in metrics:
        fig.add_trace(go.Scatter(
            x=monthly_data['period_str'],
            y=monthly_data['num_customers'],
            name='Lượt khách',
            mode='lines+markers+text',
            line=dict(color='#00CC96', width=2),
            text=monthly_data['num_customers'].apply(lambda x: f'{x:,.0f}'),
            textposition='top center',
            yaxis='y2'
        ))
    
    if 'profit' in metrics:
        fig.add_trace(go.Scatter(
            x=monthly_data['period_str'],
            y=monthly_data['gross_profit'],
            name='Lợi nhuận',
            mode='lines+markers+text',
            line=dict(color='#AB63FA', width=2),
            text=monthly_data['gross_profit'].apply(format_value),
            textposition='top center',
            yaxis='y1'
        ))
    
    fig.update_layout(
        xaxis_title=x_title,
        yaxis=dict(title="Doanh thu / Lợi nhuận (₫)", side='left'),
        yaxis2=dict(title="Lượt khách", overlaying='y', side='right'),
        height=250,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=40, b=30),
        hovermode='x unified'
    )
    
    return fig


def calculate_marketing_metrics(tours_df, start_date, end_date):
    """
    Calculate marketing and sales cost metrics (OPEX)
    """
    confirmed_tours = filter_confirmed_bookings(tours_df)
    period_tours = filter_data_by_date(confirmed_tours, start_date, end_date)
    
    total_revenue = period_tours['revenue'].sum()
    total_opex = period_tours['opex'].sum()
    total_marketing = period_tours['marketing_cost'].sum()
    total_sales = period_tours['sales_cost'].sum()
    
    opex_ratio = (total_opex / total_revenue * 100) if total_revenue > 0 else 0
    
    return {
        'total_opex': total_opex,
        'total_marketing': total_marketing,
        'total_sales': total_sales,
        'total_revenue': total_revenue,
        'opex_ratio': opex_ratio
    }


def calculate_cac_by_channel(tours_df, start_date, end_date):
    """
    Calculate Customer Acquisition Cost (CAC) by sales channel
    """
    confirmed_tours = filter_confirmed_bookings(tours_df)
    period_tours = filter_data_by_date(confirmed_tours, start_date, end_date)
    
    channel_metrics = period_tours.groupby('sales_channel').agg({
        'opex': 'sum',
        'customer_id': 'nunique',  # Unique customers
        'revenue': 'sum'
    }).reset_index()
    
    channel_metrics.columns = ['sales_channel', 'total_opex', 'unique_customers', 'revenue']
    # ĐÃ SỬA: Bảo vệ chia cho 0
    channel_metrics['cac'] = np.where(
        channel_metrics['unique_customers'] > 0,
        channel_metrics['total_opex'] / channel_metrics['unique_customers'],
        0
    )
    channel_metrics['cac'] = channel_metrics['cac'].fillna(0)
    
    return channel_metrics


def calculate_clv_by_segment(tours_df):
    """
    Calculate Customer Lifetime Value (CLV) by segment
    """
    confirmed_tours = filter_confirmed_bookings(tours_df)
    
    # Calculate CLV = Total revenue from repeat customers / Number of customers
    segment_metrics = confirmed_tours.groupby('segment').agg({
        'customer_id': 'nunique',
        'revenue': 'sum',
        'booking_id': 'count'
    }).reset_index()
    
    segment_metrics.columns = ['segment', 'unique_customers', 'total_revenue', 'total_bookings']
    # ĐÃ SỬA: Bảo vệ chia cho 0
    segment_metrics['avg_bookings_per_customer'] = np.where(
        segment_metrics['unique_customers'] > 0,
        segment_metrics['total_bookings'] / segment_metrics['unique_customers'],
        0
    )
    # ĐÃ SỬA: Bảo vệ chia cho 0
    segment_metrics['clv'] = np.where(
        segment_metrics['unique_customers'] > 0,
        segment_metrics['total_revenue'] / segment_metrics['unique_customers'],
        0
    )
    segment_metrics['clv'] = segment_metrics['clv'].fillna(0)
    
    return segment_metrics


def get_channel_breakdown(tours_df, start_date, end_date, metric='revenue'):
    """
    Get breakdown by sales channel
    """
    confirmed_tours = filter_confirmed_bookings(tours_df)
    period_tours = filter_data_by_date(confirmed_tours, start_date, end_date)
    
    if metric == 'revenue':
        channel_data = period_tours.groupby('sales_channel').agg({
            'revenue': 'sum',
            'num_customers': 'sum'
        }).reset_index()
        # ĐÃ SỬA: Bảo vệ chia cho 0
        channel_data['avg_revenue_per_customer'] = np.where(
            channel_data['num_customers'] > 0,
            channel_data['revenue'] / channel_data['num_customers'],
            0
        )
        return channel_data
    elif metric == 'customers':
        channel_data = period_tours.groupby('sales_channel').agg({
            'num_customers': 'sum',
            'revenue': 'sum' # Giữ revenue để tính Avg Rev per customer
        }).reset_index()
        # ĐÃ SỬA: Bảo vệ chia cho 0
        channel_data['avg_revenue_per_customer'] = np.where(
            channel_data['num_customers'] > 0,
            channel_data['revenue'] / channel_data['num_customers'],
            0
        )
        return channel_data
    else:  # profit
        channel_data = period_tours.groupby('sales_channel').agg({
            'gross_profit': 'sum'
        }).reset_index()
        return channel_data


def create_profit_margin_chart_with_color(data, x_col, y_col, title):
    """
    Create horizontal bar chart with continuous color scale (temperature/heatmap style)
    """
    # Use continuous color scale based on margin values
    fig = go.Figure(go.Bar(
        x=data[x_col],
        y=data[y_col],
        orientation='h',
        marker=dict(
            color=data[x_col],
            colorscale='RdYlGn',  # Red-Yellow-Green temperature scale
            showscale=True,
            colorbar=dict(
                title=dict(text="Tỷ suất LN (%)", side="right"),
                tickmode="linear",
                tick0=0,
                dtick=2,
                len=0.7
            ),
            cmin=data[x_col].min(),
            cmax=data[x_col].max()
        ),
        text=data[x_col].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        hovertemplate='%{y}<br>Tỷ suất: %{x:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Tỷ suất lợi nhuận (%)",
        yaxis_title="",
        height=max(300, len(data) * 30),
        margin=dict(l=30, r=100, t=50, b=30),
        showlegend=False
    )
    
    return fig


def get_route_detailed_table(tours_df, plans_df, start_date, end_date):
    """
    Get detailed table by route with plan comparison, including occupancy and cancellation rates.
    """
    # Lấy TẤT CẢ bookings trong kỳ (để tính hủy/đổi và tổng capacity)
    period_tours_all = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_tours = filter_confirmed_bookings(period_tours_all)
    
    if period_tours_all.empty: 
        # Cập nhật danh sách cột trả về
        return pd.DataFrame(columns=['route', 'revenue', 'num_customers', 'gross_profit', 'profit_margin', 
                                     'planned_revenue', 'revenue_completion', 'occupancy_rate', 'cancel_rate'])

    # 1. Tính ACTUALS, OCCUPANCY, và CANCEL/CHANGE Rate theo tuyến
    route_metrics = period_tours_all.groupby('route').agg(
        # Thực hiện
        revenue=('revenue', lambda x: x[period_tours_all['status'] == 'Đã xác nhận'].sum()),
        gross_profit=('gross_profit', lambda x: x[period_tours_all['status'] == 'Đã xác nhận'].sum()),
        num_customers_confirmed=('num_customers', lambda x: x[period_tours_all['status'] == 'Đã xác nhận'].sum()),
        num_customers_all=('num_customers', 'sum'),
        
        # Công suất và Hủy/Đổi
        tour_capacity=('tour_capacity', 'sum'),
        num_customers_cancelled=('num_customers', lambda x: x[x.index.isin(period_tours_all[period_tours_all['status'].isin(['Đã hủy', 'Hoãn'])].index)].sum())
    ).reset_index()
    
    # Tính Tỷ lệ Lấp đầy và Hủy/Đổi
    route_metrics['occupancy_rate'] = np.where(
        route_metrics['tour_capacity'] > 0,
        (route_metrics['num_customers_all'] / route_metrics['tour_capacity'] * 100).round(1),
        0
    )
    route_metrics['cancel_rate'] = np.where(
        route_metrics['num_customers_all'] > 0,
        (route_metrics['num_customers_cancelled'] / route_metrics['num_customers_all'] * 100).round(1),
        0
    )
    
    # 2. Xử lý Plans (Giữ nguyên logic cũ)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    plan_mask = (
        (plans_df['year'] == start_dt.year) &
        (plans_df['month'] >= start_dt.month) &
        (plans_df['month'] <= end_dt.month)
    )
    period_plans = plans_df[plan_mask]
    
    route_plan = period_plans.groupby('route').agg({
        'planned_revenue': 'sum',
        'planned_customers': 'sum',
        'planned_gross_profit': 'sum'
    }).reset_index()
    
    # 3. Merge và Final Calculation
    route_table = route_metrics.merge(route_plan, on='route', how='left').fillna(0)
    
    # Tỷ suất LN
    route_table['profit_margin'] = np.where(
        route_table['revenue'] > 0,
        (route_table['gross_profit'] / route_table['revenue'] * 100).round(1),
        0
    )

    # Tỷ lệ Hoàn thành Kế hoạch
    route_table['revenue_completion'] = np.where(
        route_table['planned_revenue'] > 0,
        (route_table['revenue'] / route_table['planned_revenue'] * 100).round(1),
        0
    )
    
    # Đổi tên cột cho phù hợp với hiển thị cũ
    route_table.rename(columns={'num_customers_confirmed': 'num_customers'}, inplace=True)
    
    # Giới hạn các cột cuối cùng (Chỉ trả về các cột cần thiết)
    return route_table[['route', 'revenue', 'num_customers', 'gross_profit', 
                        'profit_margin', 'revenue_completion', 'occupancy_rate', 'cancel_rate']]

def get_unit_detailed_table(tours_df, plans_df, start_date, end_date):
    """
    Get detailed table by business unit
    """
    confirmed_tours = filter_confirmed_bookings(tours_df)
    period_tours = filter_data_by_date(confirmed_tours, start_date, end_date)
    
    if period_tours.empty: 
        return pd.DataFrame(columns=['business_unit', 'revenue', 'num_customers', 'gross_profit', 'profit_margin', 'avg_revenue_per_customer'])

    # Actual data
    unit_actual = period_tours.groupby('business_unit').agg({
        'revenue': 'sum',
        'num_customers': 'sum',
        'gross_profit': 'sum'
    }).reset_index()
    
    # ĐÃ SỬA: Bảo vệ chia cho 0
    unit_actual['profit_margin'] = np.where(
        unit_actual['revenue'] > 0,
        (unit_actual['gross_profit'] / unit_actual['revenue'] * 100),
        0
    )
    # ĐÃ SỬA: Bảo vệ chia cho 0
    unit_actual['avg_revenue_per_customer'] = np.where(
        unit_actual['num_customers'] > 0,
        (unit_actual['revenue'] / unit_actual['num_customers']),
        0
    )
    
    # Plan data
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    plan_mask = (
        (plans_df['year'] == start_dt.year) &
        (plans_df['month'] >= start_dt.month) &
        (plans_df['month'] <= end_dt.month)
    )
    period_plans = plans_df[plan_mask]
    
    unit_plan = period_plans.groupby('business_unit').agg({
        'planned_revenue': 'sum',
        'planned_customers': 'sum',
        'planned_gross_profit': 'sum'
    }).reset_index()
    
    # Merge
    unit_table = unit_actual.merge(unit_plan, on='business_unit', how='left').fillna(0)
    
    return unit_table

def get_unit_breakdown_simple(tours_df, metric='revenue'):
    """
    Get breakdown by business unit for a specific metric (revenue/customers/profit) for pie chart.
    """
    confirmed = filter_confirmed_bookings(tours_df)
    
    if confirmed.empty:
        return pd.DataFrame(columns=['business_unit', 'value', 'percentage'])
    
    if metric == 'revenue':
        unit_data = confirmed.groupby('business_unit')['revenue'].sum().reset_index()
    elif metric == 'customers':
        unit_data = confirmed.groupby('business_unit')['num_customers'].sum().reset_index()
    else:  # profit
        unit_data = confirmed.groupby('business_unit')['gross_profit'].sum().reset_index()
        
    unit_data.columns = ['business_unit', 'value']
    
    total_value = unit_data['value'].sum()
    # Bảo vệ chia cho 0
    unit_data['percentage'] = np.where(
        total_value > 0,
        (unit_data['value'] / total_value * 100).round(1),
        0
    )
    unit_data = unit_data.sort_values('value', ascending=False)
    
    return unit_data

# --- HÀM MỚI CHO TAB 3 ---

def calculate_partner_breakdown_by_type(tours_df, status_filter):
    """Calculates active/expiring partner count broken down by partner_type."""
    # Logic này yêu cầu cột 'partner_type' phải có trong tours_df
    df_filtered = tours_df[tours_df['contract_status'] == status_filter].copy()
    
    # Định nghĩa các loại đối tác cố định để đảm bảo Expander hiển thị đủ các loại
    partner_types = ['Khách sạn', 'Ăn uống', 'Vận chuyển', 'Vé máy bay', 'Điểm tham quan', 'Đối tác nước ngoài']
    
    if df_filtered.empty:
        # Trả về DataFrame với count = 0 cho tất cả các loại
        return pd.DataFrame([{'type': t, 'count': 0} for t in partner_types])

    breakdown = df_filtered.groupby('partner_type')['partner'].nunique().reset_index()
    breakdown.columns = ['type', 'count']
    
    # Xử lý các loại đối tác không có trong dữ liệu
    existing_types = breakdown['type'].tolist()
    missing_types = [t for t in partner_types if t not in existing_types]
    
    if missing_types:
        df_missing = pd.DataFrame([{'type': t, 'count': 0} for t in missing_types])
        breakdown = pd.concat([breakdown, df_missing], ignore_index=True)
    
    return breakdown
    
def calculate_partner_performance(partner_df):
    """
    Calculates key performance metrics for partners used in the scatter plot.
    """
    # Lấy dữ liệu đã xác nhận (hoặc dữ liệu đã lọc theo kỳ)
    # Vì partner_df đã được lọc theo date/dimensional, ta dùng nó trực tiếp
    
    if partner_df.empty:
        return pd.DataFrame(columns=['partner', 'total_revenue', 'avg_feedback', 'total_customers'])

    partner_performance = partner_df.groupby('partner').agg(
        total_revenue=('revenue', 'sum'),
        # Giả định cột feedback_ratio là tỷ lệ phản hồi tích cực (0-1)
        avg_feedback=('feedback_ratio', 'mean'), 
        total_customers=('num_customers', 'sum')
    ).reset_index()
    
    # Chuyển đổi tỷ lệ phản hồi thành phần trăm
    partner_performance['avg_feedback'] = partner_performance['avg_feedback'] * 100

    return partner_performance

def calculate_partner_revenue_by_type(partner_df):
    """
    Calculates total revenue grouped by service_type for expander detail.
    """
    if partner_df.empty:
        return pd.DataFrame(columns=['service_type', 'revenue'])
    
    revenue_by_type = partner_df.groupby('service_type')['revenue'].sum().reset_index()
    revenue_by_type = revenue_by_type.sort_values('revenue', ascending=False)
    
    return revenue_by_type    

# HÀM MỚI CHO VÙNG 2 TAB 3: TÍNH TỔNG TỒN KHO DỊCH VỤ VÀ TỶ LỆ HỦY DỊCH VỤ
def calculate_service_inventory(tours_df, service_type=None):
    """Calculates total service units (customers) held by type."""
    df = tours_df.copy()
    
    if service_type and service_type != "Tất cả":
        df = df[df['service_type'] == service_type]
    
    # Tính tổng số lượng khách hàng sử dụng dịch vụ này (đơn vị tồn kho)
    inventory = df.groupby('service_type')['num_customers'].sum().reset_index()
    inventory.columns = ['service_type', 'total_units']
    return inventory

def calculate_service_cancellation_metrics(tours_df):
    """Calculates service cancellation rate based on contract status."""
    
    # Giả định: Các tour có trạng thái 'Đã hủy' hoặc 'Hoãn' liên quan đến hủy dịch vụ
    total_services = len(tours_df)
    
    if total_services == 0:
        return {'cancel_rate': 0, 'total_cancelled': 0}

    # Giả định: Hủy hợp đồng = Hủy/Hoãn Tour
    cancelled_services = tours_df[tours_df['status'].isin(['Đã hủy', 'Hoãn'])]
    total_cancelled = len(cancelled_services)
    
    cancellation_rate = (total_cancelled / total_services) * 100
    
    return {
        'cancel_rate': cancellation_rate,
        'total_cancelled': total_cancelled
    }



def calculate_partner_kpis(tours_df):
    """Calculate core KPIs for Partner Management (Vùng 1)"""
    
    # Lọc dữ liệu hợp đồng đang triển khai (Đơn giản hóa: dùng trạng thái hợp đồng)
    active_contracts = tours_df[tours_df['contract_status'].isin(["Đang triển khai", "Sắp hết hạn"])]
    
    total_active_partners = active_contracts['partner'].nunique()
    total_contracts = active_contracts['partner'].count()
    
    # Tình trạng hợp đồng
    contracts_status_count = active_contracts.groupby('contract_status')['partner'].count().reset_index()
    contracts_status_count.columns = ['status', 'count']
    
    # Tình trạng thanh toán
    payment_status_count = tours_df.groupby('payment_status')['partner'].count().reset_index()
    payment_status_count.columns = ['status', 'count']
    
    # Dịch vụ đang giữ
    service_inventory = tours_df.groupby('service_type')['num_customers'].sum().reset_index()
    service_inventory.columns = ['service_type', 'total_units']
    
    # Tính tổng doanh thu dịch vụ
    total_service_revenue = tours_df['revenue'].sum()
    
    return {
        'total_active_partners': total_active_partners,
        'total_contracts': total_contracts,
        'contracts_status_count': contracts_status_count,
        'payment_status_count': payment_status_count,
        'service_inventory': service_inventory,
        'total_service_revenue': total_service_revenue
    }

def calculate_partner_revenue_metrics(tours_df):
    """Calculate service price metrics (Vùng 2)"""
    
    # Tính giá dịch vụ (giá trung bình/khách)
    tours_df['service_price_per_pax'] = np.where(
        tours_df['num_customers'] > 0,
        tours_df['service_cost'] / tours_df['num_customers'],
        0
    )
    
    # Group by service type
    service_metrics = tours_df.groupby('service_type').agg(
        max_price=('service_price_per_pax', 'max'),
        avg_price=('service_price_per_pax', 'mean'),
        min_price=('service_price_per_pax', 'min'),
    ).reset_index()
    
    return service_metrics

def create_partner_trend_chart(tours_df, start_date, end_date):
    """Creates a combined bar/line chart for partner revenue and customer count (Vùng 3)"""
    
    period_tours = filter_data_by_date(tours_df, start_date, end_date, date_column='booking_date')
    
    # Tương tự như create_trend_chart, xác định granularity
    period_length = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    if period_length <= 60:
        freq = 'W'
        x_title = "Tuần"
        date_col = 'week_start'
        period_tours['week_start'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('W').apply(lambda x: x.start_time)
        df_trend = period_tours.groupby('week_start').agg(
            revenue=('revenue', 'sum'),
            customers=('num_customers', 'sum')
        ).reset_index()
    else:
        freq = 'M'
        x_title = "Tháng"
        date_col = 'month_start'
        period_tours['month_start'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('M').apply(lambda x: x.start_time)
        df_trend = period_tours.groupby('month_start').agg(
            revenue=('revenue', 'sum'),
            customers=('num_customers', 'sum')
        ).reset_index()
        
    if df_trend.empty:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Trace 1: Doanh thu (Cột - Trục Y chính)
    fig.add_trace(
        go.Bar(x=df_trend[date_col], y=df_trend['revenue'], name='Doanh thu Dịch vụ', marker_color='#636EFA'),
        secondary_y=False,
    )

    # Trace 2: Lượt khách (Đường - Trục Y phụ)
    fig.add_trace(
        go.Scatter(x=df_trend[date_col], y=df_trend['customers'], name='Số lượng Khách', mode='lines+markers', line=dict(color='#FFA15A', width=3)),
        secondary_y=True,
    )

    # Cập nhật layout
    fig.update_layout(
        title_text=f"Xu hướng Doanh thu và Số lượng Khách theo {x_title}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
        margin=dict(l=50, r=50, t=50, b=30),
        xaxis=dict(title=x_title),
        yaxis=dict(title="Doanh thu (₫)", side='left', showgrid=True),
        yaxis2=dict(title="Số lượng Khách", side='right', showgrid=False)
    )

    return fig

# ============================================================
# STRATEGIC FINANCIAL METRICS (TAB CHIẾN LƯỢC)
# ============================================================

def calculate_strategic_metrics(tours_df, start_date, end_date):
    """
    Calculate 10 strategic financial metrics for the Strategic tab.
    Returns a dictionary with all metrics.
    """
    filtered = filter_data_by_date(tours_df, start_date, end_date)
    confirmed = filter_confirmed_bookings(filtered)
    
    if confirmed.empty:
        return {
            'dso': 0,
            'inventory_turnover': 0,
            'breakeven_point': 0,
            'cogs_ratio': 0,
            'personnel_cost_ratio': 0,
            'sales_cost_ratio': 0,
            'opex_ratio': 0,
            'profit_margin': 0,
            'roi': 0,
            'roe': 0,
            'total_revenue': 0,
            'total_cost': 0,
            'total_profit': 0,
            'total_opex': 0
        }
    
    # Basic aggregations
    total_revenue = confirmed['revenue'].sum()
    total_cost = confirmed['cost'].sum()
    total_profit = confirmed['gross_profit'].sum()
    total_marketing = confirmed['marketing_cost'].sum()
    total_sales = confirmed['sales_cost'].sum()
    total_opex = confirmed['opex'].sum()
    
    # Assumptions for missing fields (can be adjusted based on actual data)
    # For demo purposes, we'll derive estimates from available data
    
    # 1. DSO (Days Sales Outstanding) - Kỳ thu tiền bình quân
    # Formula: (Accounts Receivable / Revenue) * Days in Period
    # Assumption: 30% of revenue is receivable (typical for tour industry)
    days_in_period = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    accounts_receivable = total_revenue * 0.3  # 30% assumption
    dso = (accounts_receivable / total_revenue * days_in_period) if total_revenue > 0 else 0
    
    # 2. Inventory Turnover - Vòng quay hàng tồn kho
    # Formula: COGS / Average Inventory
    # For service industry (tours), use capacity utilization as proxy
    total_capacity = confirmed['tour_capacity'].sum()
    total_booked = confirmed['num_customers'].sum()
    inventory_turnover = (total_booked / total_capacity) if total_capacity > 0 else 0
    
    # 3. Breakeven Point - Điểm hòa vốn (revenue needed to cover costs)
    # Formula: Fixed Costs / Contribution Margin Ratio
    # Assumption: 40% of opex is fixed, 60% variable
    fixed_costs = total_opex * 0.4
    variable_cost_ratio = (total_cost + total_opex * 0.6) / total_revenue if total_revenue > 0 else 0.8
    contribution_margin_ratio = 1 - variable_cost_ratio
    breakeven_point = fixed_costs / contribution_margin_ratio if contribution_margin_ratio > 0 else 0
    
    # 4. COGS Ratio - Tỷ lệ chi phí giá vốn trên giá bán
    cogs_ratio = (total_cost / total_revenue * 100) if total_revenue > 0 else 0
    
    # 5. Personnel Cost Ratio - Tỷ lệ chi phí nhân sự trên doanh thu
    # Assumption: 15% of OPEX is personnel cost (typical for travel agencies)
    personnel_cost = total_opex * 0.15
    personnel_cost_ratio = (personnel_cost / total_revenue * 100) if total_revenue > 0 else 0
    
    # 6. Sales Cost Ratio - Tỷ lệ chi phí bán hàng trên doanh thu
    sales_cost_ratio = (total_sales / total_revenue * 100) if total_revenue > 0 else 0
    
    # 7. Operating Expense Ratio - Tỷ lệ chi phí vận hành trên doanh thu
    opex_ratio = (total_opex / total_revenue * 100) if total_revenue > 0 else 0
    
    # 8. Profit Margin - Tỷ suất sinh lợi nhuận trên doanh thu
    profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
    
    # 9. ROI - Return on Investment
    # Formula: (Net Profit / Total Investment) * 100
    # Assumption: Total investment = COGS + OPEX
    total_investment = total_cost + total_opex
    roi = (total_profit / total_investment * 100) if total_investment > 0 else 0
    
    # 10. ROE - Return on Equity
    # Formula: (Net Profit / Equity) * 100
    # Assumption: Equity = 20% of revenue (typical capital structure)
    equity = total_revenue * 0.2
    roe = (total_profit / equity * 100) if equity > 0 else 0
    
    return {
        'dso': dso,
        'inventory_turnover': inventory_turnover * 100,  # Convert to percentage
        'breakeven_point': breakeven_point,
        'cogs_ratio': cogs_ratio,
        'personnel_cost_ratio': personnel_cost_ratio,
        'sales_cost_ratio': sales_cost_ratio,
        'opex_ratio': opex_ratio,
        'profit_margin': profit_margin,
        'roi': roi,
        'roe': roe,
        'total_revenue': total_revenue,
        'total_cost': total_cost,
        'total_profit': total_profit,
        'total_opex': total_opex,
        'personnel_cost': personnel_cost,
        'total_sales': total_sales
    }


def create_strategic_gauge(value, title, unit='%', threshold_good=None, threshold_bad=None, is_inverse=False):
    """
    Create a simplified gauge for strategic metrics.
    
    is_inverse: True if lower is better (e.g., DSO, cost ratios)
    """
    value = value if not pd.isna(value) else 0
    
    # Determine color based on thresholds
    if is_inverse:
        # Lower is better (costs, DSO)
        if threshold_good and value <= threshold_good:
            color = "#00CC96"  # Green
        elif threshold_bad and value >= threshold_bad:
            color = "#EF553B"  # Red
        else:
            color = "#FFA500"  # Orange
    else:
        # Higher is better (profits, returns, turnover)
        if threshold_good and value >= threshold_good:
            color = "#00CC96"  # Green
        elif threshold_bad and value <= threshold_bad:
            color = "#EF553B"  # Red
        else:
            color = "#FFA500"  # Orange
    
    fig = go.Figure(go.Indicator(
        mode="number",
        value=value,
        title={'text': title, 'font': {'size': 12}},
        number={'suffix': unit, 'font': {'size': 24, 'color': color}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    
    fig.update_layout(
        height=120,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    return fig


# ============================================================
# STRATEGIC METRICS CALCULATIONS
# ============================================================

def calculate_strategic_metrics(tours_df, start_date, end_date):
    """
    Calculate strategic financial metrics for the selected period.
    Returns a dictionary with all 10 strategic KPIs.
    """
    filtered_data = filter_data_by_date(tours_df, start_date, end_date)
    confirmed_data = filter_confirmed_bookings(filtered_data)
    
    if confirmed_data.empty:
        return {
            'dso': 0,
            'inventory_turnover': 0,
            'break_even_point': 0,
            'cogs_ratio': 0,
            'personnel_cost_ratio': 0,
            'sales_cost_ratio': 0,
            'operating_cost_ratio': 0,
            'profit_margin_ratio': 0,
            'roi': 0,
            'roe': 0
        }
    
    total_revenue = confirmed_data['revenue'].sum()
    total_cost = confirmed_data['cost'].sum()
    total_gross_profit = confirmed_data['gross_profit'].sum()
    total_marketing = confirmed_data['marketing_cost'].sum()
    total_sales_cost = confirmed_data['sales_cost'].sum()
    total_opex = confirmed_data['opex'].sum()
    
    # 1. DSO (Days Sales Outstanding) - Kỳ thu tiền bình quân
    # Giả định: Doanh thu/ngày = total_revenue / số ngày trong kỳ
    num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    avg_daily_revenue = total_revenue / num_days if num_days > 0 else 0
    # DSO = Accounts Receivable / Avg Daily Revenue
    # Giả định AR = 30% doanh thu (chưa thu về)
    accounts_receivable = total_revenue * 0.3
    dso = (accounts_receivable / avg_daily_revenue) if avg_daily_revenue > 0 else 0
    
    # 2. Inventory Turnover - Vòng quay hàng tồn kho
    # = COGS / Average Inventory
    # Giả định: Avg Inventory = 20% của COGS
    avg_inventory = total_cost * 0.2
    inventory_turnover = (total_cost / avg_inventory) if avg_inventory > 0 else 0
    
    # 3. Break-even Point - Điểm hòa vốn
    # = Fixed Costs / (Revenue - Variable Costs)
    # Giả định: Fixed costs = 40% OPEX, Variable costs = 60% OPEX + Cost
    fixed_costs = total_opex * 0.4
    variable_costs = total_opex * 0.6 + total_cost
    contribution_margin = total_revenue - variable_costs
    contribution_margin_ratio = (contribution_margin / total_revenue) if total_revenue > 0 else 0
    break_even_revenue = (fixed_costs / contribution_margin_ratio) if contribution_margin_ratio > 0 else 0
    break_even_point = (break_even_revenue / total_revenue * 100) if total_revenue > 0 else 0
    
    # 4. COGS Ratio - Tỷ lệ chi phí giá vốn / giá bán
    cogs_ratio = (total_cost / total_revenue * 100) if total_revenue > 0 else 0
    
    # 5. Personnel Cost Ratio - Tỷ lệ chi phí nhân sự / doanh thu
    # Giả định: Personnel = 50% của OPEX
    personnel_cost = total_opex * 0.5
    personnel_cost_ratio = (personnel_cost / total_revenue * 100) if total_revenue > 0 else 0
    
    # 6. Sales Cost Ratio - Tỷ lệ chi phí bán hàng / doanh thu
    sales_cost_ratio = (total_sales_cost / total_revenue * 100) if total_revenue > 0 else 0
    
    # 7. Operating Cost Ratio - Tỷ lệ chi phí vận hành / doanh thu
    operating_cost_ratio = (total_opex / total_revenue * 100) if total_revenue > 0 else 0
    
    # 8. Profit Margin Ratio - Tỷ suất sinh lợi nhuận / doanh thu
    net_profit = total_gross_profit - total_opex
    profit_margin_ratio = (net_profit / total_revenue * 100) if total_revenue > 0 else 0
    
    # 9. ROI - Return on Investment
    # = Net Profit / Total Investment * 100
    # Giả định: Investment = Total Cost + OPEX
    total_investment = total_cost + total_opex
    roi = (net_profit / total_investment * 100) if total_investment > 0 else 0
    
    # 10. ROE - Return on Equity
    # = Net Profit / Shareholders' Equity * 100
    # Giả định: Equity = 60% của Total Investment
    equity = total_investment * 0.6
    roe = (net_profit / equity * 100) if equity > 0 else 0
    
    return {
        'dso': dso,
        'inventory_turnover': inventory_turnover,
        'break_even_point': break_even_point,
        'cogs_ratio': cogs_ratio,
        'personnel_cost_ratio': personnel_cost_ratio,
        'sales_cost_ratio': sales_cost_ratio,
        'operating_cost_ratio': operating_cost_ratio,
        'profit_margin_ratio': profit_margin_ratio,
        'roi': roi,
        'roe': roe,
        # Raw values for display
        'total_revenue': total_revenue,
        'total_cost': total_cost,
        'total_gross_profit': total_gross_profit,
        'total_opex': total_opex,
        'net_profit': net_profit
    }


# ============================================================
# TAB 2 - BOOKING METRICS FUNCTIONS
# ============================================================

def calculate_booking_metrics(tours_df, start_date, end_date):
    """
    Calculate booking-related metrics for Tab 2
    Returns total bookings, success rate, cancellation rate, etc.
    """
    period_tours = filter_data_by_date(tours_df, start_date, end_date)
    
    if period_tours.empty:
        return {
            'total_booked_customers': 0,
            'success_rate': 0,
            'cancel_change_rate': 0
        }
    
    # Total confirmed customers
    confirmed_tours = filter_confirmed_bookings(period_tours)
    total_booked_customers = confirmed_tours['num_customers'].sum()
    
    # Success rate (confirmed bookings / total bookings)
    total_bookings = len(period_tours)
    confirmed_bookings = len(confirmed_tours)
    success_rate = (confirmed_bookings / total_bookings * 100) if total_bookings > 0 else 0
    
    # Cancellation/change rate
    cancelled_changed = len(period_tours[period_tours['status'].isin(['Đã hủy', 'Hoãn'])])
    cancel_change_rate = (cancelled_changed / total_bookings * 100) if total_bookings > 0 else 0
    
    return {
        'total_booked_customers': total_booked_customers,
        'success_rate': success_rate,
        'cancel_change_rate': cancel_change_rate
    }


def create_cancellation_trend_chart(tours_df, start_date, end_date):
    """
    Create a trend chart showing cancellation/change rate over time
    """
    period_tours = filter_data_by_date(tours_df, start_date, end_date)
    
    if period_tours.empty:
        return go.Figure().update_layout(title='Không có dữ liệu', height=300)
    
    # Determine granularity
    period_length = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    
    if period_length <= 30:
        freq = 'D'
        x_title = "Ngày"
        period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('D')
    elif period_length <= 90:
        freq = 'W'
        x_title = "Tuần"
        period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('W')
    else:
        freq = 'M'
        x_title = "Tháng"
        period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('M')
    
    # Calculate cancellation rate by period
    trend_data = period_tours.groupby('period').agg(
        total_bookings=('booking_id', 'count'),
        cancelled=('status', lambda x: x.isin(['Đã hủy', 'Hoãn']).sum())
    ).reset_index()
    
    trend_data['cancel_rate'] = np.where(
        trend_data['total_bookings'] > 0,
        (trend_data['cancelled'] / trend_data['total_bookings'] * 100),
        0
    )
    
    trend_data['period_str'] = trend_data['period'].astype(str)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_data['period_str'],
        y=trend_data['cancel_rate'],
        mode='lines+markers',
        name='Tỷ lệ Hủy/Đổi',
        line=dict(color='#EF553B', width=2)
    ))
    
    fig.update_layout(
        title='Xu hướng Tỷ lệ Hủy/Đổi Booking',
        xaxis_title=x_title,
        yaxis_title='Tỷ lệ (%)',
        height=300,
        hovermode='x unified'
    )
    
    return fig


def create_demographic_pie_chart(tours_df, column, title_suffix):
    """
    Create a pie chart for demographic breakdown (age group, nationality, etc.)
    """
    if tours_df.empty or column not in tours_df.columns:
        return go.Figure().update_layout(title=f'Không có dữ liệu {title_suffix}', height=300)
    
    demographic_data = tours_df.groupby(column)['num_customers'].sum().reset_index()
    demographic_data.columns = ['category', 'value']
    
    fig = px.pie(
        demographic_data,
        values='value',
        names='category',
        title=f'Phân bổ theo {column}',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=300, showlegend=True)
    
    return fig


def create_ratio_trend_chart(tours_df, start_date, end_date, metric='success_rate', title='Xu hướng Tỷ lệ'):
    """
    Create a trend chart for ratios over time (e.g., success rate, cancel rate)
    metric can be: 'success_rate', 'cancellation_rate', etc.
    """
    period_tours = filter_data_by_date(tours_df, start_date, end_date)
    
    if period_tours.empty:
        return go.Figure().update_layout(title='Không có dữ liệu', height=300)
    
    # Determine granularity based on period length
    period_length = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    
    if period_length <= 30:
        freq = 'D'
        x_title = "Ngày"
        period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('D')
    elif period_length <= 90:
        freq = 'W'
        x_title = "Tuần"
        period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('W')
    else:
        freq = 'M'
        x_title = "Tháng"
        period_tours['period'] = pd.to_datetime(period_tours['booking_date']).dt.to_period('M')
    
    # Calculate ratio by period based on metric type
    if metric == 'success_rate':
        trend_data = period_tours.groupby('period').agg(
            total_bookings=('booking_id', 'count'),
            target_count=('status', lambda x: x.isin(['Đã xác nhận']).sum())
        ).reset_index()
    else:  # cancellation_rate or cancel_change_rate
        trend_data = period_tours.groupby('period').agg(
            total_bookings=('booking_id', 'count'),
            target_count=('status', lambda x: x.isin(['Đã hủy', 'Hoãn']).sum())
        ).reset_index()
    
    trend_data['ratio'] = np.where(
        trend_data['total_bookings'] > 0,
        (trend_data['target_count'] / trend_data['total_bookings'] * 100),
        0
    )
    
    trend_data['period_str'] = trend_data['period'].astype(str)
    
    # Choose color based on metric
    line_color = '#00CC96' if metric == 'success_rate' else '#EF553B'
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_data['period_str'],
        y=trend_data['ratio'],
        mode='lines+markers',
        name=title,
        line=dict(color=line_color, width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title='Tỷ lệ (%)',
        height=300,
        hovermode='x unified'
    )
    
    return fig


def create_stacked_route_chart(tours_df, metric='revenue', title=''):
    """
    Create a stacked bar chart showing metric breakdown by route and business unit
    """
    if tours_df.empty:
        return go.Figure().update_layout(title='Không có dữ liệu', height=400)
    
    confirmed_tours = filter_confirmed_bookings(tours_df)
    
    # Get top 10 routes
    route_totals = confirmed_tours.groupby('route')[metric].sum().nlargest(10).index
    filtered_data = confirmed_tours[confirmed_tours['route'].isin(route_totals)]
    
    # Pivot data for stacking
    pivot_data = filtered_data.pivot_table(
        index='route',
        columns='business_unit',
        values=metric,
        aggfunc='sum',
        fill_value=0
    )
    
    fig = go.Figure()
    
    for col in pivot_data.columns:
        fig.add_trace(go.Bar(
            name=col,
            x=pivot_data.index,
            y=pivot_data[col]
        ))
    
    fig.update_layout(
        title=title or f'Top 10 Tuyến theo {metric}',
        barmode='stack',
        height=400,
        xaxis_title='Tuyến',
        yaxis_title=metric,
        showlegend=True
    )
    
    return fig


def create_top_routes_dual_axis_chart(df_merged):
    """
    Create a dual-axis chart showing revenue & profit (bars), and customers (line) for top routes
    """
    if df_merged.empty:
        return go.Figure().update_layout(title='Không có dữ liệu', height=400)
    
    # Check required columns
    required_cols = ['route', 'revenue', 'num_customers', 'gross_profit']
    for col in required_cols:
        if col not in df_merged.columns:
            return go.Figure().update_layout(title=f'Thiếu cột: {col}', height=400)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Revenue bars (left Y-axis)
    fig.add_trace(
        go.Bar(
            name='Doanh thu', 
            x=df_merged['route'], 
            y=df_merged['revenue'], 
            marker_color='#636EFA',
            text=df_merged['revenue'].apply(lambda x: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'),
            textposition='outside'
        ),
        secondary_y=False,
    )
    
    # Profit bars (left Y-axis)
    fig.add_trace(
        go.Bar(
            name='Lãi gộp', 
            x=df_merged['route'], 
            y=df_merged['gross_profit'], 
            marker_color='#00CC96',
            text=df_merged['gross_profit'].apply(lambda x: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'),
            textposition='outside'
        ),
        secondary_y=False,
    )
    
    # Customers line (right Y-axis)
    fig.add_trace(
        go.Scatter(
            name='Lượt khách', 
            x=df_merged['route'], 
            y=df_merged['num_customers'], 
            mode='lines+markers', 
            line=dict(color='#00CC96', width=3),
            marker=dict(size=10, symbol='circle'),
            text=df_merged['num_customers'].apply(lambda x: f'{x:,.0f}'),
            textposition='top center'
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        title='Top Tuyến: Doanh thu & Tỷ suất Lợi nhuận',
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode='group'
    )
    
    fig.update_xaxes(title_text='Tuyến')
    fig.update_yaxes(title_text='Doanh thu / Lãi gộp (₫)', secondary_y=False)
    fig.update_yaxes(title_text='Lượt khách', secondary_y=True)
    
    return fig


def create_top_routes_ratio_stacked(df_merged):
    """
    Create a 100% stacked bar chart showing percentage breakdown by route for Revenue, Customers, and Profit
    """
    if df_merged.empty:
        return go.Figure().update_layout(title='Không có dữ liệu', height=400)
    
    # Check required columns exist
    required_cols = ['route', 'revenue', 'num_customers', 'gross_profit']
    for col in required_cols:
        if col not in df_merged.columns:
            return go.Figure().update_layout(title=f'Thiếu cột: {col}', height=400)
    
    # Calculate percentages for each metric
    total_revenue = df_merged['revenue'].sum()
    total_customers = df_merged['num_customers'].sum()
    total_profit = df_merged['gross_profit'].sum()
    
    df_merged['revenue_pct'] = (df_merged['revenue'] / total_revenue * 100) if total_revenue > 0 else 0
    df_merged['customers_pct'] = (df_merged['num_customers'] / total_customers * 100) if total_customers > 0 else 0
    df_merged['profit_pct'] = (df_merged['gross_profit'] / total_profit * 100) if total_profit > 0 else 0
    
    fig = go.Figure()
    
    # Define color palette (match the image)
    colors = ['#FF6B6B', '#4ECDC4', '#C7B198', '#1A535C', '#FFE66D', 
              '#95E1D3', '#F38181', '#FFA07A', '#20B2AA', '#DDA15E']
    
    # Add traces for each route
    for idx, (_, row) in enumerate(df_merged.iterrows()):
        color = colors[idx % len(colors)]
        route_name = row['route']
        
        # For Revenue bar
        fig.add_trace(go.Bar(
            name=route_name,
            x=['Doanh thu'],
            y=[row['revenue_pct']],
            marker_color=color,
            text=f"{row['revenue_pct']:.1f}%",
            textposition='inside',
            showlegend=True,
            legendgroup=route_name,
            hovertemplate=f'{route_name}: %{{y:.1f}}%<extra></extra>'
        ))
        
        # For Customers bar
        fig.add_trace(go.Bar(
            name=route_name,
            x=['Lượt khách'],
            y=[row['customers_pct']],
            marker_color=color,
            text=f"{row['customers_pct']:.1f}%",
            textposition='inside',
            showlegend=False,
            legendgroup=route_name,
            hovertemplate=f'{route_name}: %{{y:.1f}}%<extra></extra>'
        ))
        
        # For Profit bar
        fig.add_trace(go.Bar(
            name=route_name,
            x=['Lợi nhuận'],
            y=[row['profit_pct']],
            marker_color=color,
            text=f"{row['profit_pct']:.1f}%",
            textposition='inside',
            showlegend=False,
            legendgroup=route_name,
            hovertemplate=f'{route_name}: %{{y:.1f}}%<extra></extra>'
        ))
    
    fig.update_layout(
        barmode='stack',
        title='Tỷ trọng Đóng góp của Top Tuyến Tour (%)',
        xaxis_title='Chỉ số',
        yaxis_title='Tỷ trọng (%)',
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02,
            title='Tuyến Tour'
        ),
        hovermode='closest'
    )
    
    return fig


def create_segment_bu_comparison_chart(df_long, grouping_col='segment'):
    """
    Create a combined chart with bars (Revenue, Profit) and line (Customers) on secondary axis
    """
    if df_long.empty:
        return go.Figure().update_layout(title='Không có dữ liệu', height=400)
    
    # Check if columns exist (case-insensitive)
    value_col = 'Value' if 'Value' in df_long.columns else 'value'
    metric_col = 'Metric' if 'Metric' in df_long.columns else 'metric'
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Get unique groups and metrics
    groups = df_long[grouping_col].unique()
    
    # Separate data for bars (Revenue, Profit) and line (Customers)
    df_revenue = df_long[df_long[metric_col] == 'Revenue']
    df_customers = df_long[df_long[metric_col] == 'Customers']
    df_profit = df_long[df_long[metric_col] == 'Profit']
    
    # Helper function to format large numbers
    def format_value(val):
        if val >= 1e9:
            return f'{val/1e9:.1f}B'
        elif val >= 1e6:
            return f'{val/1e6:.1f}M'
        elif val >= 1e3:
            return f'{val/1e3:.0f}K'
        else:
            return f'{val:.0f}'
    
    # Add Revenue bars with text labels
    if not df_revenue.empty:
        fig.add_trace(
            go.Bar(
                name='Doanh thu',
                x=df_revenue[grouping_col],
                y=df_revenue[value_col],
                marker_color='#636EFA',
                text=df_revenue[value_col].apply(format_value),
                textposition='outside'
            ),
            secondary_y=False
        )
    
    # Add Profit bars with text labels
    if not df_profit.empty:
        fig.add_trace(
            go.Bar(
                name='Lãi gộp',
                x=df_profit[grouping_col],
                y=df_profit[value_col],
                marker_color='#00CC96',
                text=df_profit[value_col].apply(format_value),
                textposition='outside'
            ),
            secondary_y=False
        )
    
    # Add Customers line on secondary axis with text labels
    if not df_customers.empty:
        fig.add_trace(
            go.Scatter(
                name='Lượt khách',
                x=df_customers[grouping_col],
                y=df_customers[value_col],
                mode='lines+markers+text',
                line=dict(color='#EF553B', width=3),
                marker=dict(size=8),
                text=df_customers[value_col].apply(lambda x: f'{x:,.0f}'),
                textposition='top center',
                textfont=dict(size=10)
            ),
            secondary_y=True
        )
    
    # Update layout
    fig.update_xaxes(title_text=grouping_col.capitalize())
    fig.update_yaxes(title_text="Doanh thu / Lãi gộp (₫)", secondary_y=False)
    fig.update_yaxes(title_text="Lượt khách", secondary_y=True)
    
    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        barmode='group'
    )
    
    return fig


def calculate_partner_service_usage_by_period(tours_df, start_date, end_date):
    """
    Calculate partner service usage over time periods (series/allotment tracking)
    Returns: DataFrame with partner, period, total_allocated, used, unused, cancelled
    """
    # Filter data by date
    df = tours_df.copy()
    df['booking_date'] = pd.to_datetime(df['booking_date'])
    df = df[(df['booking_date'] >= pd.to_datetime(start_date)) & 
            (df['booking_date'] <= pd.to_datetime(end_date))]
    
    # Calculate period length to determine granularity
    period_length = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    
    if period_length <= 7:
        df['period'] = df['booking_date'].dt.to_period('D')
        df['period_str'] = df['period'].dt.strftime('%d/%m')
    elif period_length <= 60:
        df['period'] = df['booking_date'].dt.to_period('W')
        df['period_str'] = df['period'].apply(lambda x: f"T{x.week}")
    else:
        df['period'] = df['booking_date'].dt.to_period('M')
        df['period_str'] = df['period'].astype(str)
    
    # Calculate service usage by partner and period
    # Assumption: tour_capacity = allocated units (series/allotment contracted)
    # num_customers = used units
    # cancelled = tours with status 'Đã hủy' or 'Hoãn'
    
    result = []
    for partner in df['partner'].unique():
        partner_df = df[df['partner'] == partner]
        
        for period in partner_df['period_str'].unique():
            period_df = partner_df[partner_df['period_str'] == period]
            
            total_allocated = period_df['tour_capacity'].sum()  # Total contracted capacity
            used = period_df[period_df['status'] == 'Xác nhận']['num_customers'].sum()
            cancelled = period_df[period_df['status'].isin(['Đã hủy', 'Hoãn'])]['num_customers'].sum()
            unused = total_allocated - used - cancelled
            
            result.append({
                'partner': partner,
                'period': period,
                'total_allocated': total_allocated,
                'used': used,
                'unused': max(0, unused),
                'cancelled': cancelled,
                'service_type': period_df['service_type'].mode()[0] if len(period_df) > 0 else ''
            })
    
    return pd.DataFrame(result)


def create_partner_service_usage_chart(df_usage):
    """
    Create stacked bar chart showing service usage by partner over time
    Shows: Used, Unused, Cancelled for each partner across periods
    """
    if df_usage.empty:
        return go.Figure().update_layout(title='Không có dữ liệu', height=400)
    
    # Group by partner and aggregate all periods
    partner_summary = df_usage.groupby('partner').agg({
        'total_allocated': 'sum',
        'used': 'sum',
        'unused': 'sum',
        'cancelled': 'sum'
    }).reset_index()
    
    # Sort by total allocated descending
    partner_summary = partner_summary.sort_values('total_allocated', ascending=False).head(10)
    
    fig = go.Figure()
    
    # Add traces for each status
    fig.add_trace(go.Bar(
        name='Đã sử dụng',
        x=partner_summary['partner'],
        y=partner_summary['used'],
        marker_color='#00CC96',
        text=partner_summary['used'].apply(lambda x: f'{x:,.0f}'),
        textposition='inside'
    ))
    
    fig.add_trace(go.Bar(
        name='Chưa sử dụng',
        x=partner_summary['partner'],
        y=partner_summary['unused'],
        marker_color='#FFA15A',
        text=partner_summary['unused'].apply(lambda x: f'{x:,.0f}'),
        textposition='inside'
    ))
    
    fig.add_trace(go.Bar(
        name='Đã hủy',
        x=partner_summary['partner'],
        y=partner_summary['cancelled'],
        marker_color='#EF553B',
        text=partner_summary['cancelled'].apply(lambda x: f'{x:,.0f}'),
        textposition='inside'
    ))
    
    fig.update_layout(
        title='Tổng số lượng Dịch vụ và Tình trạng Sử dụng theo Đối tác',
        xaxis_title='Đối tác',
        yaxis_title='Số lượng (khách)',
        barmode='stack',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    
    return fig


def calculate_service_utilization_by_bu(tours_df):
    """
    Calculate service utilization rate by business unit
    Formula: (Used Services / Total Contracted Services) x 100%
    Returns: DataFrame with business_unit, utilization_rate for TreeMap
    """
    df = tours_df.copy()
    
    # Calculate by business unit
    bu_summary = df.groupby('business_unit').agg({
        'tour_capacity': 'sum',  # Total contracted capacity (allocated)
        'num_customers': 'sum'   # Total used
    }).reset_index()
    
    # Calculate utilization rate
    bu_summary['total_contracted'] = bu_summary['tour_capacity']
    bu_summary['total_used'] = bu_summary['num_customers']
    bu_summary['utilization_rate'] = (bu_summary['total_used'] / bu_summary['total_contracted'] * 100).fillna(0)
    bu_summary['utilization_rate'] = bu_summary['utilization_rate'].clip(0, 100)
    
    # Add label for display
    bu_summary['label'] = bu_summary.apply(
        lambda row: f"{row['business_unit']}<br>{row['utilization_rate']:.1f}%<br>({row['total_used']:,.0f}/{row['total_contracted']:,.0f})",
        axis=1
    )
    
    return bu_summary


def create_service_utilization_treemap(df_bu_util):
    """
    Create TreeMap showing service utilization rate by business unit
    Color intensity based on utilization rate
    """
    if df_bu_util.empty:
        return go.Figure().update_layout(title='Không có dữ liệu', height=400)
    
    fig = go.Figure(go.Treemap(
        labels=df_bu_util['business_unit'],
        parents=[''] * len(df_bu_util),
        values=df_bu_util['total_contracted'],
        text=df_bu_util['label'],
        textposition='middle center',
        marker=dict(
            colorscale='RdYlGn',  # Red (low) -> Yellow -> Green (high)
            cmid=50,
            colorbar=dict(title="Tỷ lệ sử dụng (%)", ticksuffix="%"),
            cmin=0,
            cmax=100,
            line=dict(width=2, color='white')
        ),
        marker_colorscale='RdYlGn',
        marker_colors=df_bu_util['utilization_rate'],
        hovertemplate='<b>%{label}</b><br>Tỷ lệ sử dụng: %{color:.1f}%<br>Tổng contracted: %{value:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Chỉ số Sử dụng Kho Dịch vụ theo Đơn vị Kinh doanh (%)',
        height=400,
        margin=dict(t=50, l=10, r=10, b=10)
    )
    
    return fig