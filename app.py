"""
Vietravel Business Intelligence Dashboard
Comprehensive tour sales performance, revenue, profit margins, and operational metrics dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz # Cần thiết cho Timezone handling
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
# Cần import make_subplots ở đây để dùng trong app.py nếu cần cho chart phức tạp
from plotly.subplots import make_subplots 
from admin_ui import render_admin_ui

# Import custom modules
from data_generator import load_or_generate_data
from utils import (
    # Các hàm Format và Core Logic
    format_currency, format_number, format_percentage,
    calculate_completion_rate, get_growth_rate, filter_data_by_date, filter_confirmed_bookings,
    
    # Các hàm KPI và Chart
    calculate_kpis, 
    create_gauge_chart, create_bar_chart, create_pie_chart, create_line_chart,
    
    # Các hàm Top/Breakdown
    get_top_routes, get_route_unit_breakdown, get_unit_breakdown,
    get_segment_breakdown, get_segment_unit_breakdown, get_channel_breakdown,
    get_unit_breakdown_simple,
    
    # Các hàm Operational và Detailed Tables
    calculate_operational_metrics, get_low_margin_tours, get_unit_performance, 
    get_route_detailed_table, get_unit_detailed_table,
    
    # Các hàm Marketing/CLV/Forecast
    create_forecast_chart, create_trend_chart, 
    calculate_marketing_metrics, calculate_cac_by_channel, calculate_clv_by_segment, 
    create_profit_margin_chart_with_color,
    calculate_partner_performance,
    
    # Các hàm Đối tác mới (ĐÃ THÊM)
    calculate_partner_kpis, calculate_partner_revenue_metrics, create_partner_trend_chart,
    calculate_partner_breakdown_by_type,calculate_service_inventory, calculate_service_cancellation_metrics,
    calculate_partner_revenue_by_type,
    calculate_partner_service_usage_by_period, create_partner_service_usage_chart,
    calculate_service_utilization_by_bu, create_service_utilization_treemap,
    
    # Các hàm Chiến lược (TAB 4)
    calculate_strategic_metrics, create_strategic_gauge,
    
    # Các hàm Tab 2 - Booking Metrics
    calculate_booking_metrics, create_cancellation_trend_chart, create_demographic_pie_chart,
    create_ratio_trend_chart, create_stacked_route_chart, create_top_routes_dual_axis_chart,
    create_top_routes_ratio_stacked, create_segment_bu_comparison_chart
)

# Page configuration
st.set_page_config(
    page_title="Vietravel BI Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to reduce padding and whitespace
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    h1 {
        padding-top: 0rem;
        margin-top: 0rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding-top: 8px;
        padding-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Nhập nguồn dữ liệu (đặt trước khi load dữ liệu)
with st.sidebar:
    st.markdown("---")
    st.subheader("Nguồn dữ liệu")
    use_sheet = st.checkbox("Dùng Google Sheet (CSV public)", value=st.session_state.get('use_sheet', False))
    sheet_url = st.text_input(
        "Link Google Sheet",
        value=st.session_state.get('sheet_url', ''),
        help="Dán link Google Sheet (bấm Share → Anyone with the link → Viewer). Có thể giữ #gid hiện tại."
    )
    # Lưu lại vào session_state để sử dụng khi load
    st.session_state['use_sheet'] = use_sheet
    st.session_state['sheet_url'] = sheet_url

# Initialize session state for data
# Load data when not already loaded or when explicitly requested (data_loaded flag False)
if not st.session_state.get('data_loaded', False):
    with st.spinner('Đang tải dữ liệu...'):
        # load_or_generate_data now returns (tours_df, plans_df, historical_df, meta)
        spreadsheet_url = st.session_state.get('sheet_url') if st.session_state.get('use_sheet') else None
        result = load_or_generate_data(spreadsheet_url)
        # Support both old and new signatures for safety
        if isinstance(result, tuple) and len(result) == 4:
            tours_df, plans_df, historical_df, data_meta = result
        else:
            tours_df, plans_df, historical_df = result
            data_meta = {'used_excel': False, 'processed_files': [], 'parsed_rows': 0}

        st.session_state.tours_df = tours_df
        st.session_state.plans_df = plans_df
        st.session_state.historical_df = historical_df
        st.session_state.data_meta = data_meta
        st.session_state.data_loaded = True

    # Show a banner if data was loaded from external source
    meta = st.session_state.get('data_meta', {})
    if meta.get('used_excel') or meta.get('used_sheet'):
        files = st.session_state['data_meta'].get('processed_files', [])
        parsed = st.session_state['data_meta'].get('parsed_rows', 0)
        files_str = ', '.join(files) if files else '(<no filenames>)'
        st.sidebar.success(f"Dữ liệu được tải từ Google Sheet/Excel: {files_str} — {parsed} dòng parsed")

# Load data from session state
tours_df = st.session_state.tours_df
plans_df = st.session_state.plans_df
historical_df = st.session_state.historical_df

# Dashboard Title
st.title("📊 VIETRAVEL - DASHBOARD KINH DOANH TOUR")

# Sidebar filters
with st.sidebar:
    st.header("🔍 Bộ lọc dữ liệu")
    
    # Date range selector
    st.subheader("Khoảng thời gian")
    
    # Quick date range options
    date_option = st.selectbox(
        "Chọn kỳ báo cáo",
        ["Tuần", "Tháng", "Quý", "Năm", "Tùy chỉnh"]
    )
    
    # Xử lý Timezone an toàn
    vietnam_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    today = datetime.now(vietnam_tz).replace(tzinfo=None) # Naive datetime
    
    if date_option == "Tuần":
        # 7 ngày gần nhất
        start_date = today - timedelta(days=6)
        start_date = datetime(start_date.year, start_date.month, start_date.day)
        end_date = today
    elif date_option == "Tháng":
        # Tháng hiện tại
        start_date = datetime(today.year, today.month, 1)
        end_date = today
    elif date_option == "Quý":
        # Quý hiện tại
        quarter = (today.month - 1) // 3 + 1
        start_date = datetime(today.year, 3 * quarter - 2, 1)
        end_date = today
    elif date_option == "Năm":
        # Năm hiện tại
        start_date = datetime(today.year, 1, 1)
        end_date = today
    else:  # Tùy chỉnh
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Từ ngày",
                value=datetime(today.year, today.month, 1)
            )
        with col2:
            end_date = st.date_input(
                "Đến ngày",
                value=today
            )
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
    
    st.markdown(f"**Kỳ báo cáo:** {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
    
    # Business unit filter
    st.subheader("Đơn vị kinh doanh")
    business_units = ["Tất cả"] + sorted(tours_df['business_unit'].unique().tolist())
    selected_unit = st.selectbox("Chọn đơn vị", business_units)
    
    # Route filter
    st.subheader("Tuyến tour")
    if selected_unit != "Tất cả":
        routes = ["Tất cả"] + sorted(
            tours_df[tours_df['business_unit'] == selected_unit]['route'].unique().tolist()
        )
    else:
        routes = ["Tất cả"] + sorted(tours_df['route'].unique().tolist())
    selected_route = st.selectbox("Chọn tuyến", routes)
    
    # Segment filter
    st.subheader("Phân khúc")
    segments = ["Tất cả"] + sorted(tours_df['segment'].unique().tolist())
    selected_segment = st.selectbox("Chọn phân khúc", segments)
    
    # Top N selector
    st.subheader("Thiết lập hiển thị")
    top_n = st.slider("Top N tuyến tour", min_value=5, max_value=15, value=10)
    
    # Bổ sung Filter cho Tab 3
    st.markdown("---")
    st.subheader("Bộ lọc Đối tác")
    partners = ["Tất cả"] + sorted(tours_df['partner'].unique().tolist())
    selected_partner = st.selectbox("Chọn Đối tác", partners)
    
    service_types = ["Tất cả"] + sorted(tours_df['service_type'].unique().tolist())
    selected_service = st.selectbox("Chọn Loại dịch vụ", service_types)

    st.markdown("---")
    
    # Refresh data button
    if st.button("🔄 Làm mới dữ liệu", width='stretch'):
        st.session_state.data_loaded = False
        st.rerun()

# Filter data based on selections (dimensional filters only, NOT date)
# Date filtering will be done inside calculate_kpis to preserve YoY data
tours_filtered_dimensional = tours_df.copy()
filtered_plans = plans_df.copy()

if selected_unit != "Tất cả":
    tours_filtered_dimensional = tours_filtered_dimensional[tours_filtered_dimensional['business_unit'] == selected_unit]
    filtered_plans = filtered_plans[filtered_plans['business_unit'] == selected_unit]

if selected_route != "Tất cả":
    tours_filtered_dimensional = tours_filtered_dimensional[tours_filtered_dimensional['route'] == selected_route]
    filtered_plans = filtered_plans[filtered_plans['route'] == selected_route]

if selected_segment != "Tất cả":
    tours_filtered_dimensional = tours_filtered_dimensional[tours_filtered_dimensional['segment'] == selected_segment]
    filtered_plans = filtered_plans[filtered_plans['segment'] == selected_segment]

# Áp dụng bộ lọc đối tác cho Tab 3
partner_filtered_df = tours_filtered_dimensional.copy()
if selected_partner != "Tất cả":
    partner_filtered_df = partner_filtered_df[partner_filtered_df['partner'] == selected_partner]
if selected_service != "Tất cả":
    partner_filtered_df = partner_filtered_df[partner_filtered_df['service_type'] == selected_service]

# Calculate KPIs using dimensionally filtered data (calculate_kpis will handle date filtering)
kpis = calculate_kpis(tours_filtered_dimensional, filtered_plans, start_date, end_date)


# Also create a date+dimension filtered version for charts that don't need historical data
filtered_tours = filter_data_by_date(tours_filtered_dimensional, start_date, end_date)

# TÍNH TOÁN BOOKING METRICS CHO TAB 2 (ĐÃ DI CHUYỂN)
booking_metrics = calculate_booking_metrics(tours_df, start_date, end_date)


if 'show_admin_ui' not in st.session_state:
    st.session_state.show_admin_ui = False

# Nút mở/đóng UI Admin (đặt ở khu vực trên cùng)
col_toggle, col_empty = st.columns([1, 4])

with col_toggle:
    if st.session_state.show_admin_ui:
        if st.button("<< Quay lại Dashboard Chính", type="secondary"):
            st.session_state.show_admin_ui = False
            st.rerun()
    else:
        if st.button("🔧 Mở UI Nhập liệu/Sửa Hợp đồng (Admin)", type="secondary"):
            st.session_state.show_admin_ui = True
            st.rerun()

# ----------------------------------------------------
# KHU VỰC HIỂN THỊ UI ADMIN LỚN
# ----------------------------------------------------
if st.session_state.show_admin_ui:
    render_admin_ui() # <--- GỌI HÀM TỪ FILE admin_ui.py







# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard theo dõi Kinh Doanh",
    "🔍 Dashboard theo dõi sản phẩm",
    "🤝 Dashboard theo dõi Đối tác",
    "🎯 Chiến lược" 
])

# ============================================================
# TAB 1: TỔNG QUAN (5 VÙNG THEO SPEC)
# ============================================================
with tab1:
    # ========== VÙNG 1: TỐC ĐỘ ĐẠT KẾ HOẠCH ==========
    st.markdown("### Vùng 1: Tốc độ đạt Kế hoạch")
    
    # Row: 3 Gauge charts + 1 Forecast chart
    col1, col2, col3 = st.columns(3)
    
    # Get unit breakdown data for hover tooltips
    revenue_breakdown = get_unit_breakdown(filtered_tours, filtered_plans, start_date, end_date, metric='revenue')
    profit_breakdown = get_unit_breakdown(filtered_tours, filtered_plans, start_date, end_date, metric='profit')
    customers_breakdown = get_unit_breakdown(filtered_tours, filtered_plans, start_date, end_date, metric='customers')
    
    with col1:
        fig_revenue = create_gauge_chart(
            kpis['revenue_completion'],
            "Đạt KH Doanh thu",
            unit_breakdown=revenue_breakdown,
            actual=kpis['actual_revenue'],
            planned=kpis['planned_revenue']
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        profit_completion = calculate_completion_rate(kpis['actual_gross_profit'], kpis['planned_gross_profit'])
        fig_profit = create_gauge_chart(
            profit_completion,
            "Đạt KH Lãi Gộp",
            unit_breakdown=profit_breakdown,
            actual=kpis['actual_gross_profit'],
            planned=kpis['planned_gross_profit']
        )
        st.plotly_chart(fig_profit, use_container_width=True)
    
    with col3:
        fig_customers = create_gauge_chart(
            kpis['customer_completion'],
            "Đạt KH Lượt khách",
            unit_breakdown=customers_breakdown,
            actual=kpis['actual_customers'],
            planned=kpis['planned_customers']
        )
        st.plotly_chart(fig_customers, use_container_width=True)
    
# ========== BIỂU ĐỒ DỰ BÁO HOÀN THÀNH KẾ HOẠCH (SỬA LỖI 4 ĐỐI SỐ) ==========
# Hàng 2: Tiến độ KH theo Khu vực (1 cột) | Dự báo Hoàn thành KH (2 cột)
    st.markdown("#### Phân tích Tiến độ & Dự báo")
    col1, col2 = st.columns([1, 2]) # Tỉ lệ 1:2
    
    # Lấy dữ liệu cần thiết cho Hàng 2
    unit_performance = get_unit_performance(tours_filtered_dimensional, filtered_plans, start_date, end_date)
    
    with col1:
        st.markdown("##### 📊 Tiến độ KH theo Khu vực")
        if not unit_performance.empty:
            fig = go.Figure()
            colors = ['#00CC96' if x >= 100 else '#FFA500' if x >= 80 else '#EF553B' 
                        for x in unit_performance['revenue_completion']]
            customdata = [[row['actual_revenue'], row['planned_revenue'], row['revenue_completion']]
                          for _, row in unit_performance.iterrows()]
            fig.add_trace(go.Bar(
                x=unit_performance['business_unit'],
                y=unit_performance['revenue_completion'],
                text=[f"{v:.1f}%" for v in unit_performance['revenue_completion']],
                textposition='outside',
                marker_color=colors,
                customdata=customdata,
                hovertemplate='<b>%{x}</b><br>DT thực hiện: %{customdata[0]:,.0f} ₫<br>DT kế hoạch: %{customdata[1]:,.0f} ₫<br>Tiến độ: %{customdata[2]:.1f}%<extra></extra>'
            ))
            fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="KH 100%")
            fig.update_layout(xaxis_title="", yaxis_title="Tiến độ (%)", height=300, showlegend=False, margin=dict(l=30, r=30, t=10, b=30))
            st.plotly_chart(fig)
        else:
            st.info("Không có dữ liệu tiến độ cho khu vực kinh doanh được chọn.")
    
    with col2:
        st.markdown("##### 📈 Dự báo Hoàn thành Kế hoạch")
        fig_forecast = create_forecast_chart(
            filtered_tours, 
            filtered_plans, 
            start_date, 
            end_date,
            date_option
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.markdown("---")
    


    # ========== VÙNG 2: CHỈ SỐ TỔNG QUAN ==========
    st.markdown("###  Vùng 2: Các Chỉ số")
    
    # Row 1: 3 KPI Cards 
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="💰 DOANH THU TỔNG",
            value=format_currency(kpis['actual_revenue']),
            delta=f"{format_percentage(kpis['revenue_growth'])} so với cùng kỳ"
        )
        with st.expander("Chi tiết"):
            st.write(f"**Kế hoạch:** {format_currency(kpis['planned_revenue'])}")
            st.write(f"**Thực hiện:** {format_currency(kpis['actual_revenue'])}")
            st.write(f"**Hoàn thành:** {format_percentage(kpis['revenue_completion'])}")
            st.write(f"**Cùng kỳ năm trước:** {format_currency(kpis['ly_revenue'])}")
            st.write(f"**Tăng trưởng:** {format_percentage(kpis['revenue_growth'])}")
    
    with col2:
        st.metric(
            label="💵 Lãi Gộp",
            value=format_currency(kpis['actual_gross_profit']),
            delta=f"{format_percentage(kpis['profit_growth'])} so với cùng kỳ"
        )
        with st.expander("Chi tiết"):
            st.write(f"**Kế hoạch:** {format_currency(kpis['planned_gross_profit'])}")
            st.write(f"**Thực hiện:** {format_currency(kpis['actual_gross_profit'])}")
            profit_completion = calculate_completion_rate(kpis['actual_gross_profit'], kpis['planned_gross_profit'])
            st.write(f"**Hoàn thành:** {format_percentage(profit_completion)}")
            st.write(f"**Cùng kỳ năm trước:** {format_currency(kpis['ly_gross_profit'])}")
            st.write(f"**Tăng trưởng:** {format_percentage(kpis['profit_growth'])}")
    
    with col3:
        st.metric(
            label="👥 LƯỢT KHÁCH TỔNG",
            value=format_number(kpis['actual_customers']),
            delta=f"{format_percentage(kpis['customer_growth'])} so với cùng kỳ"
        )
        with st.expander("Chi tiết"):
            st.write(f"**Kế hoạch:** {format_number(kpis['planned_customers'])}")
            st.write(f"**Thực hiện:** {format_number(kpis['actual_customers'])}")
            st.write(f"**Hoàn thành:** {format_percentage(kpis['customer_completion'])}")
            st.write(f"**Cùng kỳ năm trước:** {format_number(kpis['ly_customers'])}")
            st.write(f"**Tăng trưởng:** {format_percentage(kpis['customer_growth'])}")
    
    # Row 2: Marketing/Sales Cost and Trend Chart
    st.markdown("")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Calculate marketing metrics
        marketing_metrics = calculate_marketing_metrics(filtered_tours, start_date, end_date)
        st.metric(
            label="💳 CHI PHÍ MARKETING/BÁN HÀNG",
            value=f"{format_percentage(marketing_metrics['opex_ratio'])}",
            delta=f"{format_currency(marketing_metrics['total_opex'])} OPEX"
        )
        with st.expander("Chi tiết"):
            st.write(f"**Chi phí Marketing:** {format_currency(marketing_metrics['total_marketing'])}")
            st.write(f"**Chi phí Bán hàng:** {format_currency(marketing_metrics['total_sales'])}")
            st.write(f"**Tổng OPEX:** {format_currency(marketing_metrics['total_opex'])}")
            st.write(f"**Doanh thu:** {format_currency(marketing_metrics['total_revenue'])}")
            st.write(f"**Tỷ lệ OPEX/DT:** {format_percentage(marketing_metrics['opex_ratio'])}")
    
    with col2:
        st.markdown("<div style='font-size: 14px; font-weight: bold; margin-bottom: 10px;'>📊 Xu hướng Doanh thu / Lượt khách / Lãi Gộp theo thời gian</div>", unsafe_allow_html=True)
        fig_trend = create_trend_chart(filtered_tours, start_date, end_date, metrics=['revenue', 'customers', 'profit'])
        st.plotly_chart(fig_trend, use_container_width=True)

    # Row 3 (MỚI): Doanh thu trung bình/Khách (AOV) + Dòng tiền thu theo ngày
    st.markdown("")
    col1, col2 = st.columns([1, 2]) # Vẫn dùng tỉ lệ 1:2 để căn chỉnh

    # Tính toán AOV
    aov = kpis['actual_revenue'] / kpis['actual_customers'] if kpis['actual_customers'] > 0 else 0
    ly_aov = kpis['ly_revenue'] / kpis['ly_customers'] if kpis['ly_customers'] > 0 else 0
    aov_growth = get_growth_rate(aov, ly_aov)

    with col1:
        st.metric(
            label="💵 DOANH THU TB/KHÁCH (AOV)",
            value=format_currency(aov),
            delta=f"{format_percentage(aov_growth)} so với cùng kỳ"
        )
        with st.expander("Chi tiết"):
            st.write(f"**AOV Cùng kỳ:** {format_currency(ly_aov)}")
            st.write(f"**Tăng trưởng AOV:** {format_percentage(aov_growth)}")
            st.write(f"**Doanh thu Tổng:** {format_currency(kpis['actual_revenue'])}")
            st.write(f"**Lượt khách Tổng:** {format_number(kpis['actual_customers'])}")

    # Col 2 (trống để căn chỉnh)
    with col2:
        st.empty()
    st.markdown("---")
    
    
    # ========== VÙNG MỚI: DÒNG TIỀN VÀ XU HƯỚNG THEO NGÀY ==========
    st.markdown("### Vùng 3: Dòng tiền và Xu hướng bán hàng theo ngày")
    
    col1, col2 = st.columns(2)
    
    # Biểu đồ 1: Dòng tiền thu theo ngày (Bar Chart)
    with col1:
        st.markdown("##### 💵 Dòng tiền thu theo ngày")
        daily_df = filtered_tours.copy()
        if not daily_df.empty:
            daily_df['date'] = pd.to_datetime(daily_df['booking_date']).dt.date
            daily_rev = daily_df.groupby('date', as_index=False)['revenue'].sum()

            fig_cash = go.Figure(go.Bar(
                x=daily_rev['date'],
                y=daily_rev['revenue'],
                marker_color='#00CC96',
                text=daily_rev['revenue'].apply(lambda x: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'),
                textposition='outside'
            ))
            fig_cash.update_traces(hovertemplate='%{x}<br>Thu: %{y:,.0f} ₫<extra></extra>')
            fig_cash.update_layout(
                height=350,
                margin=dict(l=30, r=30, t=10, b=50),
                xaxis_title="",
                yaxis_title="Doanh thu (₫)",
                showlegend=False
            )
            st.plotly_chart(fig_cash, use_container_width=True)
        else:
            st.info("Không có dữ liệu trong kỳ để hiển thị dòng tiền.")
    
    # Biểu đồ 2: Xu hướng các tuyến bán theo ngày (Line Chart)
    with col2:
        st.markdown("##### 📈 Xu hướng các tuyến bán theo ngày (Doanh số)")
        daily_route_df = filtered_tours.copy()
        if not daily_route_df.empty:
            daily_route_df['date'] = pd.to_datetime(daily_route_df['booking_date']).dt.date
            
            # Lấy top 5 tuyến theo doanh thu tổng
            top_routes_list = filtered_tours.groupby('route')['revenue'].sum().nlargest(5).index.tolist()
            daily_route_filtered = daily_route_df[daily_route_df['route'].isin(top_routes_list)]
            
            daily_route_rev = daily_route_filtered.groupby(['date', 'route'], as_index=False)['revenue'].sum()
            
            fig_route_trend = px.line(
                daily_route_rev, 
                x='date', 
                y='revenue', 
                color='route',
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_route_trend.update_traces(hovertemplate='%{x}<br>%{fullData.name}<br>DT: %{y:,.0f} ₫<extra></extra>')
            fig_route_trend.update_layout(
                height=350,
                margin=dict(l=30, r=30, t=10, b=50),
                xaxis_title="",
                yaxis_title="Doanh thu (₫)",
                legend=dict(title="Tuyến", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            st.plotly_chart(fig_route_trend, use_container_width=True)
        else:
            st.info("Không có dữ liệu tuyến để hiển thị xu hướng.")
    
    st.markdown("---")
    
    
    # ========== VÙNG 4: PHÂN THEO PHÂN KHÚC & ĐƠN VỊ KINH DOANH ==========
    st.markdown("### Vùng 4: Phân theo Phân khúc & Đơn vị Kinh doanh")
    SEGMENT_COLORS = ['#3CB371', '#6495ED', '#FFA07A']
    BU_COLORS = ['#3CB371', '#6495ED', '#FFA07A', '#FF6347']
    
    # --- HÀNG 1: PHÂN TÍCH THEO PHÂN KHÚC (BAR CHART NHÓM) ---
    st.markdown("#### Hàng 1: Hiệu suất theo Phân khúc (FIT / GIT / Inbound)")
    col1, col2 = st.columns(2)
    
    # 1. Chuẩn bị dữ liệu cho Phân khúc (Revenue, Customers, Profit)
    segment_revenue = get_segment_breakdown(filtered_tours, start_date, end_date, metric='revenue')
    segment_customers = get_segment_breakdown(filtered_tours, start_date, end_date, metric='customers')
    segment_profit = get_segment_breakdown(filtered_tours, start_date, end_date, metric='profit')

    # Gom dữ liệu Phân khúc
    df_segment_comp = segment_revenue[['segment', 'value']].rename(columns={'value': 'Revenue'}).merge(
        segment_customers[['segment', 'value']].rename(columns={'value': 'Customers'}), on=['segment'], how='outer'
    ).merge(
        segment_profit[['segment', 'value']].rename(columns={'value': 'Profit'}), on=['segment'], how='outer'
    ).fillna(0)
    
    # Chuyển sang định dạng long
    df_segment_long = pd.melt(df_segment_comp, id_vars=['segment'], 
                              value_vars=['Revenue', 'Customers', 'Profit'], 
                              var_name='Metric', value_name='Value')

    with col1:
        st.markdown("##### 📈 So sánh DT, LK, LN theo Phân khúc")
        fig_segment_bar = create_segment_bu_comparison_chart(df_segment_long, grouping_col='segment') # Hàm mới
        fig_segment_bar.update_layout(height=350)
        st.plotly_chart(fig_segment_bar, use_container_width=True)
        
    with col2:
        st.markdown("##### Phân bố Doanh thu ")
        # Vẫn giữ 1 Pie Chart Doanh thu để xem tỷ trọng (%)
        if not segment_revenue.empty:
            fig = go.Figure(go.Pie(
                labels=segment_revenue['segment'],
                values=segment_revenue['value'],
                textinfo='label+percent',
                marker=dict(colors=SEGMENT_COLORS)
            ))
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
            st.plotly_chart(fig)


    st.markdown("---")
    
    # --- HÀNG 2: PHÂN TÍCH THEO KHU VỰC (BAR CHART NHÓM) ---
    st.markdown("#### Hàng 2: Hiệu suất theo Khu vực Đơn vị Kinh doanh")
    
    # 2. Chuẩn bị dữ liệu cho Đơn vị Kinh doanh
    bu_revenue = get_unit_breakdown_simple(filtered_tours, metric='revenue').rename(columns={'value': 'Revenue', 'business_unit': 'group'})
    bu_customers = get_unit_breakdown_simple(filtered_tours, metric='customers').rename(columns={'value': 'Customers', 'business_unit': 'group'})
    bu_profit = get_unit_breakdown_simple(filtered_tours, metric='profit').rename(columns={'value': 'Profit', 'business_unit': 'group'})
    
    # Gom dữ liệu Đơn vị Kinh doanh
    df_bu_comp = bu_revenue[['group', 'Revenue']].merge(
        bu_customers[['group', 'Customers']], on='group', how='inner'
    ).merge(
        bu_profit[['group', 'Profit']], on='group', how='inner'
    )
    
    df_bu_long = pd.melt(df_bu_comp, id_vars=['group'], 
                              value_vars=['Revenue', 'Customers', 'Profit'], 
                              var_name='Metric', value_name='Value')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📈 So sánh DT, LK, LN theo Khu vực")
        fig_bu_bar = create_segment_bu_comparison_chart(df_bu_long, grouping_col='group') # Hàm mới
        fig_bu_bar.update_layout(height=350)
        st.plotly_chart(fig_bu_bar, use_container_width=True)
        
    with col2:
        st.markdown("##### Phân bố Doanh thu Khu vực")
        if not bu_revenue.empty:
            fig = go.Figure(go.Pie(
                labels=bu_revenue['group'],
                values=bu_revenue['Revenue'],
                textinfo='label+percent',
                marker=dict(colors=BU_COLORS)
            ))
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
            st.plotly_chart(fig)
    
    st.markdown("---")

    
    # ========== VÙNG 5: THEO ĐƠN VỊ KINH DOANH ==========
    st.markdown("### Vùng 5: Hiệu suất theo Đơn vị Kinh doanh")
    
    # Get unit data
    unit_table = get_unit_detailed_table(filtered_tours, filtered_plans, start_date, end_date)
    
    # Row 1: Revenue vs Plan comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### So sánh Doanh thu Thực hiện và Kế hoạch")
        if not unit_table.empty:
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
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=unit_table['business_unit'],
                y=unit_table['planned_revenue'],
                name='Kế hoạch',
                marker_color='#FFA15A',
                text=unit_table['planned_revenue'].apply(format_value),
                textposition='outside'
            ))
            fig.add_trace(go.Bar(
                x=unit_table['business_unit'],
                y=unit_table['revenue'],
                name='Thực hiện',
                marker_color='#636EFA',
                text=unit_table['revenue'].apply(format_value),
                textposition='outside'
            ))
            fig.update_layout(xaxis_title="", yaxis_title="Doanh thu (₫)", height=300, barmode='group', margin=dict(l=30, r=30, t=10, b=80))
            st.plotly_chart(fig)
    
    with col2:
        st.markdown("#### Tỷ suất Lãi Gộp theo Đơn vị")
        if not unit_table.empty:
            unit_margin = unit_table[['business_unit', 'profit_margin']].copy()
            fig = create_profit_margin_chart_with_color(unit_margin, 'profit_margin', 'business_unit', '')
            st.plotly_chart(fig)
    
    # Row 2: Detailed table
    st.markdown("#### Bảng số liệu chi tiết theo Đơn vị")
    if not unit_table.empty:
        display_df = unit_table.copy()
        display_df = display_df[[
            'business_unit', 'revenue', 'num_customers', 'gross_profit',
            'profit_margin', 'avg_revenue_per_customer'
        ]]
        display_df['revenue'] = display_df['revenue'].apply(format_currency)
        display_df['num_customers'] = display_df['num_customers'].apply(format_number)
        display_df['gross_profit'] = display_df['gross_profit'].apply(format_currency)
        display_df['profit_margin'] = display_df['profit_margin'].apply(lambda x: f"{x:.1f}%")
        display_df['avg_revenue_per_customer'] = display_df['avg_revenue_per_customer'].apply(format_currency)
        display_df.columns = ['Đơn vị', 'Doanh thu', 'Lượt khách', 'Lãi Gộp', 'Tỷ suất LN (%)', 'DT TB/khách']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
# ========== VÙNG 6: THÔNG TIN TUYẾN TOUR ==========
    st.markdown("### Vùng 6: Thông tin tuyến tour")

    # Chuẩn bị dữ liệu cho cả 3 chỉ số
    top_revenue = get_top_routes(filtered_tours, n=10, metric='revenue')
    top_customers = get_top_routes(filtered_tours, n=10, metric='customers')
    top_profit = get_top_routes(filtered_tours, n=10, metric='profit')

    # Hợp nhất dữ liệu Top 10 vào 1 DataFrame duy nhất để so sánh
    df_merged_top10 = pd.DataFrame({'route': top_revenue['route'].tolist()})
    df_merged_top10 = df_merged_top10.merge(top_revenue[['route', 'revenue', 'profit_margin']], on='route', how='left')
    df_merged_top10 = df_merged_top10.merge(top_customers[['route', 'num_customers']], on='route', how='left')
    df_merged_top10 = df_merged_top10.merge(top_profit[['route', 'gross_profit']], on='route', how='left')
    df_merged_top10 = df_merged_top10.fillna(0)
    df_merged_top10 = df_merged_top10.sort_values('revenue', ascending=False) # Sắp xếp theo DT

    # --- HÀNG 1: BIỂU ĐỒ 1 - SO SÁNH TUYỆT ĐỐI (TRỤC KÉP) ---
    st.markdown("#### Hàng 1: So sánh Giá trị Tuyệt đối (Doanh thu, Lượt khách, Lãi Gộp)")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📈 So sánh DT, LK, LN theo Tuyến Tour")
        if not df_merged_top10.empty:
            # Hàm mới: Biểu đồ cột nhóm/kết hợp với trục kép
            fig_dual_axis = create_top_routes_dual_axis_chart(df_merged_top10) # <--- Hàm mới
            st.plotly_chart(fig_dual_axis, use_container_width=True)
        else:
            st.info("Không có dữ liệu Top 10 Tuyến Tour.")

    # --- HÀNG 2: BIỂU ĐỒ 2 - TỶ TRỌNG ĐÓNG GÓP (100% STACKED PIE/BAR) ---
    with col2:
        st.markdown("##### 📊 Tỷ trọng Đóng góp của Top 10 Tuyến Tour")
        if not df_merged_top10.empty:
            # Hàm mới: Biểu đồ cột xếp chồng 100% cho Tỷ trọng DT, LK, LN
            fig_stacked_ratio = create_top_routes_ratio_stacked(df_merged_top10) # <--- Hàm mới
            st.plotly_chart(fig_stacked_ratio, use_container_width=True)
        else:
            st.info("Không có dữ liệu tỷ trọng.")


    st.markdown("---")
    
    # ========== VÙNG 7: CHỈ SỐ QUẢN LÝ HOẠT ĐỘNG ==========
    st.markdown("### Vùng 7: Chỉ số Quản lý Hoạt động")
    
    # Calculate operational metrics (use all-time dimensional data for accurate rates)
    ops_metrics = calculate_operational_metrics(tours_filtered_dimensional)
    
    # Row: 3 Operational gauge charts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_occ = create_gauge_chart(
            ops_metrics['avg_occupancy'],
            "Tỷ lệ Lấp đầy BQ",
            max_value=100,
            threshold=75
        )
        st.plotly_chart(fig_occ, key="gauge_tab1")
    
    with col2:
        fig_cancel = create_gauge_chart(
            ops_metrics['cancel_rate'],
            "Tỷ lệ Khách Hủy/Hoãn",
            max_value=30,
            threshold=10,
            is_inverse_metric=True
        )
        st.plotly_chart(fig_cancel)
    
    with col3:
        fig_return = create_gauge_chart(
            ops_metrics['returning_rate'],
            "Tỷ lệ Khách Quay lại",
            max_value=100,
            threshold=30
        )
        st.plotly_chart(fig_return)


# ============================================================
# TAB 2: CHI TIẾT (3 VÙNG THEO SPEC)
# ============================================================
with tab2:
    route_table = get_route_detailed_table(filtered_tours, filtered_plans, start_date, end_date)
    top_revenue = get_top_routes(filtered_tours, n=10, metric='revenue')
    top_customers = get_top_routes(filtered_tours, n=10, metric='customers')
    top_profit = get_top_routes(filtered_tours, n=10, metric='profit')
# ========== VÙNG 1: TÓM TẮT HIỆU SUẤT BOOKING (ĐÃ THÊM KPI VÀ TRENDS) ==========
    st.markdown("### Vùng 1: Tóm tắt Hiệu suất Booking")
    
    # --- Hàng 1: KPI Cấp cao ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👥 Số lượng khách đã đặt",
            value=format_number(booking_metrics['total_booked_customers'])
        )

    with col2:
        st.metric(
            label="💰 Tổng Doanh thu",
            value=format_currency(kpis['actual_revenue'])
        )
    with col3:
        st.markdown("##### 📈 Tỷ lệ Lấp đầy BQ")
        fig_occ = create_gauge_chart(
            ops_metrics['avg_occupancy'],
            "Tỷ lệ Lấp đầy BQ",
            max_value=100, 
            threshold=75,
            is_inverse_metric=False
        )
        st.plotly_chart(fig_occ, use_container_width=True, key="gauge_tab2")
    with col4:
        st.empty()

    st.markdown("---")


    # --- Hàng 2: Tỷ lệ Thành công (Gauge & Trend) ---
    st.markdown("#### 🟢 Hiệu suất Booking Thành công")
    col1, col2 = st.columns([1, 3]) # Tỷ lệ 1:3 cho Gauge và Line Chart

    with col1:
        # Tỷ lệ booking thành công (Gauge Chart)
        fig_success = create_gauge_chart(
            booking_metrics['success_rate'],
            "Tỷ lệ booking thành công",
            max_value=100, 
            threshold=90
        )
        st.plotly_chart(fig_success, use_container_width=True)
    
    with col2:
        # Xu hướng tỷ lệ booking thành công (Line Chart)
        fig_success_trend = create_ratio_trend_chart(tours_df, start_date, end_date, 
                                                     metric='success_rate', 
                                                     title='Xu hướng Tỷ lệ Booking Thành công (Theo ngày/tuần)')
        st.plotly_chart(fig_success_trend, use_container_width=True)

    st.markdown("---")


    # --- Hàng 3: Tỷ lệ Hủy/Đổi (Gauge & Trend) ---
    st.markdown("#### 🔴 Hiệu suất Khách Hủy/Đổi")
    col1, col2 = st.columns([1, 3]) # Tỷ lệ 1:3 cho Gauge và Line Chart

    with col1:
        # Tỷ lệ khách hàng hủy tour hoặc thay đổi (Gauge Chart)
        fig_cancel = create_gauge_chart(
            booking_metrics['cancel_change_rate'],
            "Tỷ lệ Khách Hủy/Đổi",
            max_value=30, 
            threshold=15, 
            is_inverse_metric=True
        )
        st.plotly_chart(fig_cancel, use_container_width=True)
        
    with col2:
        # Xu hướng tỷ lệ khách hàng hủy tour (Line Chart)
        fig_cancel_trend_ratio = create_ratio_trend_chart(tours_df, start_date, end_date, 
                                                           metric='cancellation_rate', 
                                                           title='Xu hướng Tỷ lệ Khách Hủy/Đổi (Theo ngày/tuần)')
        st.plotly_chart(fig_cancel_trend_ratio, use_container_width=True)

    st.markdown("---")


    # ========== VÙNG 2: THEO TUYẾN ==========
    st.markdown("### Vùng 2: Phân tích theo Tuyến")
    
    # Get route data
    route_table = get_route_detailed_table(filtered_tours, filtered_plans, start_date, end_date)
    top_revenue = get_top_routes(filtered_tours, n=10, metric='revenue')
    top_customers = get_top_routes(filtered_tours, n=10, metric='customers')
    top_profit = get_top_routes(filtered_tours, n=10, metric='profit')
    
    # Row 1: Top tuyến Tour charts
    st.markdown("#### Top Tuyến Tour")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Doanh thu (Phân bổ BU)")
        fig_rev_stacked = create_stacked_route_chart(filtered_tours, metric='revenue', title='')
        st.plotly_chart(fig_rev_stacked, use_container_width=True)
    
    with col2:
        st.markdown("##### Lượt khách (Phân bổ BU)")
        fig_cust_stacked = create_stacked_route_chart(filtered_tours, metric='num_customers', title='')
        st.plotly_chart(fig_cust_stacked, use_container_width=True)
    
    with col3:
        st.markdown("##### Lãi Gộp (Phân bổ BU)")
        fig_profit_stacked = create_stacked_route_chart(filtered_tours, metric='gross_profit', title='')
        st.plotly_chart(fig_profit_stacked, use_container_width=True)
    
    st.markdown("")

    # Row 2: Profit margin with color coding
    st.markdown("#### Tỷ suất Lãi Gộp theo Tuyến")
    if not route_table.empty:
        top_10_margin = route_table.nlargest(10, 'profit_margin')[['route', 'profit_margin']]
        fig = create_profit_margin_chart_with_color(top_10_margin, 'profit_margin', 'route', '')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

    # Row 3: Detailed table
    st.markdown("#### Bảng số liệu chi tiết theo Tuyến")
    if not route_table.empty:
        display_df = route_table.copy()
        display_df = display_df[[
            'route', 'revenue', 'num_customers', 'gross_profit', 
            'profit_margin', 'revenue_completion', 'occupancy_rate', 'cancel_rate'
        ]]
        display_df['revenue'] = display_df['revenue'].apply(format_currency)
        display_df['num_customers'] = display_df['num_customers'].apply(format_number)
        display_df['gross_profit'] = display_df['gross_profit'].apply(format_currency)
        display_df['profit_margin'] = display_df['profit_margin'].apply(lambda x: f"{x:.1f}%")
        display_df['revenue_completion'] = display_df['revenue_completion'].apply(lambda x: f"{x:.1f}%")
        display_df['occupancy_rate'] = display_df['occupancy_rate'].apply(lambda x: f"{x:.1f}%")
        display_df['cancel_rate'] = display_df['cancel_rate'].apply(lambda x: f"{x:.1f}%")
        display_df.columns = ['Tuyến', 'Doanh thu', 'Lượt khách', 'Lãi Gộp', 
                      'Tỷ suất LN (%)', 'Tiến độ KH (%)', 'Tỷ lệ Lấp đầy (%)', 'Tỷ lệ Hủy/Đổi (%)']

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("")
    

    
    # ========== VÙNG 3: THEO KÊNH BÁN VÀ PHÂN KHÚC ==========
    st.markdown("### Vùng 3: Theo Kênh bán và Phân khúc")
    
    # Get channel and segment data
    channel_revenue = get_channel_breakdown(filtered_tours, start_date, end_date, metric='revenue')
    channel_customers = get_channel_breakdown(filtered_tours, start_date, end_date, metric='customers')
    segment_revenue = get_segment_breakdown(filtered_tours, start_date, end_date, metric='revenue')
    segment_customers = get_segment_breakdown(filtered_tours, start_date, end_date, metric='customers')
    cac_data = calculate_cac_by_channel(filtered_tours, start_date, end_date)
    clv_data = calculate_clv_by_segment(tours_filtered_dimensional)
    
    # Row 1: Kênh bán pie charts
    st.markdown("#### Phân bố theo Kênh bán")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Doanh thu")
        if not channel_revenue.empty:
            fig = create_pie_chart(channel_revenue, 'revenue', 'sales_channel', '')
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)
    
    with col2:
        st.markdown("##### Lượt khách")
        if not channel_customers.empty:
            fig = create_pie_chart(channel_customers, 'num_customers', 'sales_channel', '')
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)
    
    with col3:
        st.markdown("##### Doanh thu TB/khách")
        if not channel_revenue.empty:
            fig = go.Figure(go.Bar(
                x=channel_revenue['sales_channel'],
                y=channel_revenue['avg_revenue_per_customer'],
                text=[format_currency(v) for v in channel_revenue['avg_revenue_per_customer']],
                textposition='outside',
                marker_color='#636EFA'
            ))
            fig.update_layout(xaxis_title="Doanh thu TB/khách (₫)", yaxis_title="", height=200, showlegend=False, margin=dict(l=30, r=30, t=10, b=60))
            st.plotly_chart(fig)
    
    # Row 2: Kênh bán detailed table
    if not channel_revenue.empty:
        display_df = channel_revenue.copy()
        display_df['revenue'] = display_df['revenue'].apply(format_currency)
        display_df['num_customers'] = display_df['num_customers'].apply(format_number)
        display_df['avg_revenue_per_customer'] = display_df['avg_revenue_per_customer'].apply(format_currency)
        display_df.columns = ['Kênh bán', 'Doanh thu', 'Lượt khách', 'Doanh thu TB/khách']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("")
    
    # Row 3: Phân khúc pie charts
    st.markdown("#### Phân bố theo Phân khúc")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Doanh thu")
        if not segment_revenue.empty:
            fig = create_pie_chart(segment_revenue, 'value', 'segment', '')
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)
    
    with col2:
        st.markdown("##### Lượt khách")
        if not segment_customers.empty:
            fig = create_pie_chart(segment_customers, 'value', 'segment', '')
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)
    
    st.markdown("")
    
    # Row 4: CAC and CLV
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Chi phí Thu hút Khách hàng (CAC) theo Kênh")
        if not cac_data.empty:
            fig = go.Figure(go.Bar(
                y=cac_data['sales_channel'],
                x=cac_data['cac'],
                orientation='h',
                text=[format_currency(v) for v in cac_data['cac']],
                textposition='outside',
                marker_color='#FFA15A'
            ))
            fig.update_layout(xaxis_title="CAC (₫)", yaxis_title="", height=200, showlegend=False, margin=dict(l=100, r=100, t=10, b=30))
            st.plotly_chart(fig)
    
    with col2:
        st.markdown("#### Giá trị Trọn đời Khách hàng (CLV) theo Phân khúc")
        if not clv_data.empty:
            fig = go.Figure(go.Bar(
                y=clv_data['segment'],
                x=clv_data['clv'],
                orientation='h',
                text=[format_currency(v) for v in clv_data['clv']],
                textposition='outside',
                marker_color='#00CC96'
            ))
            fig.update_layout(xaxis_title="CLV (₫)", yaxis_title="", height=200, showlegend=False, margin=dict(l=100, r=100, t=10, b=30))
            st.plotly_chart(fig)
    
    st.markdown("---")

# ========== VÙNG 4: XU HƯỚNG VÀ NHÂN KHẨU HỌC (MỚI) ==========
    st.markdown("### Vùng 4: Xu hướng và Nhân khẩu học")

    # Hàng 1: 2 Biểu đồ Xu hướng (Revenue Trend, Cancellation Trend)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Xu hướng Doanh thu theo thời kỳ")
        # Xu hướng doanh thu theo từng thời kỳ (Line Chart)
        fig_rev_trend = create_trend_chart(filtered_tours, start_date, end_date, metrics=['revenue'])
        st.plotly_chart(fig_rev_trend, use_container_width=True)
        
    with col2:
        st.markdown("##### Xu hướng Khách hàng hủy/đổi tour")
        # Xu hướng khách hàng hủy tour (Line Chart)
        fig_cancel_trend = create_cancellation_trend_chart(tours_df, start_date, end_date)
        st.plotly_chart(fig_cancel_trend, use_container_width=True)

    # Hàng 2: 2 Biểu đồ Tỷ trọng (Age, Nationality)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Tỷ trọng Doanh thu theo Độ tuổi")
        # Tỷ trọng doanh thu khách hàng theo độ tuổi (Pie Chart)
        # Giả định cột customer_age_group tồn tại
        fig_age_pie = create_demographic_pie_chart(filtered_tours, 'customer_age_group', '')
        st.plotly_chart(fig_age_pie, use_container_width=True)

    with col2:
        st.markdown("##### Tỷ trọng Doanh thu theo Quốc tịch")
        # Tỷ trọng doanh thu khách hàng theo quốc tịch (Pie Chart)
        # Giả định cột customer_nationality tồn tại
        fig_nat_pie = create_demographic_pie_chart(filtered_tours, 'customer_nationality', '')
        st.plotly_chart(fig_nat_pie, use_container_width=True)
        
    st.markdown("---")





# ============================================================
# TAB 3: ĐỐI TÁC (TÁI CẤU TRÚC HOÀN CHỈNH)
# ============================================================
with tab3:
    st.title("🤝 Dashboard Quản lý Dịch vụ và Đối tác")
    
    # Lấy dữ liệu đã lọc theo Đối tác/Dịch vụ
    # Giả định các hàm tính toán đã được định nghĩa trong utils.py hoặc được import
    partner_filtered_data = filter_data_by_date(partner_filtered_df, start_date, end_date)
    partner_kpis = calculate_partner_kpis(partner_filtered_data)
    partner_revenue_metrics = calculate_partner_revenue_metrics(partner_filtered_data)
    service_cancel_metrics = calculate_service_cancellation_metrics(partner_filtered_data)
    service_inventory_total = calculate_service_inventory(partner_filtered_data)['total_units'].sum()
    partner_performance = calculate_partner_performance(partner_filtered_data) 
    
    # Dữ liệu phân tích chi tiết theo loại (cho Expander Vùng 1)
    active_breakdown = calculate_partner_breakdown_by_type(partner_filtered_data, status_filter="Đang triển khai")
    expiring_breakdown = calculate_partner_breakdown_by_type(partner_filtered_data, status_filter="Sắp hết hạn")
    
    # --- VÙNG 1: TỔNG QUAN KPIs VÀ CẢNH BÁO (ĐÃ THÊM CHI TIẾT DỊCH VỤ) ---
    st.markdown("### 🎯 Vùng 1: Tổng quan Đối tác & Cảnh báo Hợp đồng")
    
    # Hàng 1: 4 KPI Cards tập trung
    col1, col2, col3, col4 = st.columns(4)
    
    # Tổng đối tác Đang triển khai
    with col1:
        st.metric(
            label="🤝 Tổng đối tác Đang triển khai",
            delta=" Tăng 2",
            value=format_number(partner_kpis['total_active_partners'])
        )
        # THÊM CHI TIẾT: Phân theo Loại Dịch vụ
        with st.expander("Chi tiết: Đang triển khai"):
            for _, row in active_breakdown.iterrows():
                st.write(f"**{row['type']}**: {format_number(row['count'])} đối tác")
        
    # Hợp đồng Sắp hết hạn (Cảnh báo)
    with col2:
        expiring_contracts = partner_kpis['contracts_status_count'][partner_kpis['contracts_status_count']['status'] == 'Sắp hết hạn']['count'].sum()
        st.metric(
            label="🚨 Hợp đồng Sắp hết hạn",
            value=format_number(expiring_contracts),
            delta="Cần gia hạn",
            delta_color="inverse"
        )
        # THÊM CHI TIẾT: Phân theo Loại Dịch vụ
        with st.expander("Chi tiết: Sắp hết hạn"):
            for _, row in expiring_breakdown.iterrows():
                st.write(f"**{row['type']}**: {format_number(row['count'])} hợp đồng")
        
    # Tổng Doanh thu dịch vụ (Revenue)
    with col3:
        st.metric(
            label="💰 Tổng Dịch vụ đang giữ",
            delta=" Tăng 2 tỷ",
            value=format_currency(partner_kpis['total_service_revenue'])
        )
        # THÊM CHI TIẾT: Phân theo Loại Dịch vụ
        # Giả định hàm calculate_partner_revenue_by_type trả về DataFrame: type, revenue
        revenue_by_type = calculate_partner_revenue_by_type(partner_filtered_data) # <--- Cần hàm này trong utils.py
        with st.expander("Chi tiết: Doanh thu theo Loại DV"):
            for _, row in revenue_by_type.iterrows():
                st.write(f"**{row['service_type']}**: {format_currency(row['revenue'])}")
        
    # Tình trạng Hủy dịch vụ (Gauge Chart)
    with col4:
        st.markdown("##### Tỷ lệ Hủy Dịch vụ")
        fig_service_cancel = create_gauge_chart(
            service_cancel_metrics['cancel_rate'],
            "Tỷ lệ Hủy Dịch vụ",
            max_value=30, 
            threshold=10, 
            is_inverse_metric=True
        )
        st.plotly_chart(fig_service_cancel, use_container_width=True)

    st.markdown("---")
    
    
    # --- VÙNG 2: PHÂN TÍCH TÌNH TRẠNG HỢP ĐỒNG & PHÂN TÍCH DỊCH VỤ (ĐÃ SỬA CHÚ THÍCH) ---
    st.markdown("### 📊 Vùng 2: Trạng thái Hợp đồng & Phân tích Dịch vụ")
    
    # Dữ liệu cho biểu đồ tròn (Tỷ trọng Trả trước/Trả sau)
    payment_status_data = partner_filtered_data.groupby('payment_status')['partner'].count().reset_index()
    payment_status_data.columns = ['status', 'count']
    
    col_status, col_price = st.columns([1, 2])
    
    # 1. Biểu đồ: Tỷ trọng Trạng thái Thanh toán (Pie Chart)
    with col_status:
        st.markdown("##### Tỷ trọng Thanh toán Hợp đồng")
        payment_data = payment_status_data[payment_status_data['status'].isin(['Trả trước', 'Trả sau'])].copy()
        total_payment_contracts = payment_data['count'].sum() # TỔNG HỢP ĐỒNG
        
        if not payment_data.empty:
            count_prepaid = payment_data[payment_data['status'] == 'Trả trước']['count'].iloc[0] if 'Trả trước' in payment_data['status'].values else 0
            count_postpaid = payment_data[payment_data['status'] == 'Trả sau']['count'].iloc[0] if 'Trả sau' in payment_data['status'].values else 0
            
            # --- HIỂN THỊ CHÚ THÍCH MỚI ---
            st.markdown(f"""
            <div style="font-size: 14px; font-weight: bold; text-align: center; margin-bottom: 5px;">
                Tổng Hợp đồng: {format_number(total_payment_contracts)}
            </div>
            <div style="font-size: 13px; text-align: center; margin-bottom: 5px;">
                <span style="color: #636EFA;">■ Trả trước:</span> {format_number(count_prepaid)} hợp đồng
                <span style="color: #FFA15A; margin-left: 15px;">■ Trả sau:</span> {format_number(count_postpaid)} hợp đồng
            </div>
            """, unsafe_allow_html=True)
            
            # --- TẠO BIỂU ĐỒ TRÒN (TẮT CHÚ THÍCH TỰ ĐỘNG) ---
            fig_payment_pie = px.pie(
                payment_data, 
                values='count', 
                names='status',
                color_discrete_sequence=['#636EFA', '#FFA15A'],
            )
            
            fig_payment_pie.update_traces(textinfo='percent+label', 
                                            hovertemplate='<b>%{label}</b><br>Số lượng: %{value:,.0f}<br>Tỉ lệ: %{percent}<extra></extra>')
            
            fig_payment_pie.update_layout(
                height=300, # Đã chỉnh height thấp hơn
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False
            )
            
            st.plotly_chart(fig_payment_pie, use_container_width=True)
        else:
            st.info("Không có dữ liệu hợp đồng Trả trước/Trả sau.")
            
        # Thống kê chi tiết
        active_breakdown = calculate_partner_breakdown_by_type(partner_filtered_data, status_filter="Đang triển khai")
        with st.expander("Phân loại Đối tác Đang triển khai"):
             for _, row in active_breakdown.iterrows():
                 st.write(f"**{row['type']}**: {format_number(row['count'])} đối tác")

    # 2. Bar Chart: Giá Dịch vụ (Giá TB/Khách)
    with col_price:
        st.markdown("##### Phân tích Giá Dịch vụ (Max, Avg, Min)")
        if not partner_revenue_metrics.empty:
            df_melted = partner_revenue_metrics.melt(
                id_vars='service_type',
                value_vars=['max_price', 'avg_price', 'min_price'],
                var_name='price_type',
                value_name='price_value'
            )
            
            df_melted['price_type'] = df_melted['price_type'].replace({
                'max_price': 'Giá Cao nhất',
                'avg_price': 'Giá Trung bình',
                'min_price': 'Giá Thấp nhất'
            })
            
            fig_price_comp = px.bar(
                df_melted,
                x='price_value',
                y='service_type',
                color='price_type',
                orientation='h',
                title='Giá Dịch vụ theo Loại (Max, Avg, Min)',
                barmode='group'
            )
            fig_price_comp.update_xaxes(title="Giá (₫)")
            fig_price_comp.update_traces(hovertemplate='%{x:,.0f} ₫<extra></extra>')
            fig_price_comp.update_layout(height=350, yaxis={'categoryorder':'total ascending'}, margin=dict(t=30))
            st.plotly_chart(fig_price_comp, use_container_width=True)
        
    st.markdown("---")


    # --- VÙNG 3: XU HƯỚNG VÀ HIỆU QUẢ HỢP TÁC ---
    st.markdown("### 📈 Vùng 3: Xu hướng và Hiệu quả Hợp tác")
    
    # Row 1: Biểu đồ Doanh thu và Số lượng khách theo thời gian
    col_trend, col_scatter = st.columns(2)
    
    with col_trend:
        st.markdown("##### Xu hướng Doanh thu và Lượt khách từ Đối tác")
        fig_partner_trend = create_partner_trend_chart(partner_filtered_df, start_date, end_date)
        st.plotly_chart(fig_partner_trend, use_container_width=True)
    
    with col_scatter:
        st.markdown("##### Đánh giá Hiệu quả Từng Đối tác")
        if not partner_performance.empty:
            # Biểu đồ Bong bóng: X=Doanh thu, Y=Tỷ lệ Phản hồi, Size=Số lượng khách
            fig_scatter = px.scatter(
                partner_performance,
                x='total_revenue',
                y='avg_feedback',
                size='total_customers',
                color='partner',
                hover_name='partner',
                title='Hiệu quả Đối tác (DT vs Phản hồi Tích cực)',
                labels={'total_revenue': 'Doanh thu (₫)', 'avg_feedback': 'Tỷ lệ phản hồi tích cực (%)', 'total_customers': 'Lượt khách'}
            )
            fig_scatter.update_traces(hovertemplate='<b>%{hovertext}</b><br>Doanh thu: %{x:,.0f} ₫<br>Phản hồi: %{y:.1%}<br>Lượt khách: %{marker.size:,.0f}<extra></extra>')
            fig_scatter.update_layout(height=400, showlegend=False, margin=dict(t=30))
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    
    # Row 2: NEW - Service Usage & Utilization
    col_usage, col_util = st.columns(2)
    
    with col_usage:
        st.markdown("##### Doanh thu từ Đối tác theo Thời kỳ và Tình trạng Sử dụng")
        # Calculate service usage by period
        df_service_usage = calculate_partner_service_usage_by_period(partner_filtered_df, start_date, end_date)
        fig_usage = create_partner_service_usage_chart(df_service_usage)
        st.plotly_chart(fig_usage, use_container_width=True)
    
    with col_util:
        st.markdown("##### Chỉ số Sử dụng Kho Dịch vụ theo Đơn vị")
        # Calculate utilization by business unit
        df_bu_util = calculate_service_utilization_by_bu(partner_filtered_df)
        fig_treemap = create_service_utilization_treemap(df_bu_util)
        st.plotly_chart(fig_treemap, use_container_width=True)

    # Bảng chi tiết Doanh thu/Chi phí/Lãi Gộp
    st.markdown("#### Bảng Chi tiết Hợp đồng và Tỷ suất Lãi Gộp")
    
    # Lấy bảng hợp đồng chi tiết
    df_partner_revenue_detail = partner_filtered_data.groupby(['partner', 'service_type', 'payment_status', 'contract_status']).agg(
        total_revenue=('revenue', 'sum'),
        total_service_cost=('service_cost', 'sum'),
        num_bookings=('booking_id', 'count')
    ).reset_index()
    
    df_partner_revenue_detail['profit_margin'] = np.where(
        df_partner_revenue_detail['total_revenue'] > 0,
        ((df_partner_revenue_detail['total_revenue'] - df_partner_revenue_detail['total_service_cost']) / df_partner_revenue_detail['total_revenue']) * 100,
        0
    )
    
    # Áp dụng formatting
    df_partner_revenue_detail['total_revenue'] = df_partner_revenue_detail['total_revenue'].apply(format_currency)
    df_partner_revenue_detail['total_service_cost'] = df_partner_revenue_detail['total_service_cost'].apply(format_currency)
    df_partner_revenue_detail['profit_margin'] = df_partner_revenue_detail['profit_margin'].apply(lambda x: f"{x:.1f}%")

    df_partner_revenue_detail.rename(columns={
        'contract_status': 'Trạng thái HĐ', 
        'service_type': 'Loại DV', 
        'payment_status': 'Tình trạng TT', 
        'total_revenue': 'Doanh thu',
        'total_service_cost': 'Chi phí DV',
        'num_bookings': 'SL HĐ',
        'profit_margin': 'Tỷ suất LN (%)'
    }, inplace=True)
    
    # Hàm highlight_expiring (Giữ nguyên)
    def highlight_expiring(s):
        if s['Trạng thái HĐ'] == 'Sắp hết hạn':
            return ['background-color: #ffe0e0; color: red'] * len(s)
        return [''] * len(s)

    st.dataframe(
        df_partner_revenue_detail[['partner', 'Loại DV', 'Doanh thu', 'Chi phí DV', 'Tỷ suất LN (%)', 'Trạng thái HĐ', 'Tình trạng TT']]
        .style.apply(highlight_expiring, axis=1), 
        use_container_width=True, hide_index=True
    )

st.markdown("---")

# ============================================================
# TAB 4: CHIẾN LƯỢC (10 CHỈ SỐ TÀI CHÍNH CHIẾN LƯỢC)
# ============================================================
with tab4:
    st.title("🎯 Dashboard Chiến lược")
    
    # Calculate strategic metrics
    strategic = calculate_strategic_metrics(tours_filtered_dimensional, start_date, end_date)
    
    # ========== VÙNG 1: THANH KHOẢN & HIỆU QUẢ ==========
    st.markdown("### 📊 Vùng 1: Thanh khoản & Hiệu quả Vận hành")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1. Kỳ Thu tiền Bình quân (DSO)")
        fig_dso = create_strategic_gauge(
            strategic['dso'], 
            "Days Sales Outstanding",
            unit=" ngày",
            threshold_good=30,
            threshold_bad=60,
            is_inverse=True
        )
        st.plotly_chart(fig_dso, use_container_width=True)
        with st.expander("📖 Giải thích"):
            st.write(f"""
            **DSO (Days Sales Outstanding)** đo lường số ngày trung bình để thu được tiền từ khách hàng sau khi bán tour.
            
            - **Giá trị hiện tại:** {strategic['dso']:.1f} ngày
            - **Chuẩn ngành:** 30-45 ngày
            - **Ý nghĩa:** DSO thấp = thu tiền nhanh = dòng tiền tốt
            """)
    
    with col2:
        st.markdown("#### 2. Vòng quay Hàng tồn kho")
        fig_inv = create_strategic_gauge(
            strategic['inventory_turnover'],
            "Inventory Turnover",
            unit="%",
            threshold_good=70,
            threshold_bad=50,
            is_inverse=False
        )
        st.plotly_chart(fig_inv, use_container_width=True)
        with st.expander("📖 Giải thích"):
            st.write(f"""
            **Vòng quay hàng tồn kho** (cho ngành tour: tỷ lệ lấp đầy chỗ) đo hiệu quả sử dụng capacity.
            
            - **Giá trị hiện tại:** {strategic['inventory_turnover']:.1f}%
            - **Mục tiêu:** ≥ 70%
            - **Ý nghĩa:** Tỷ lệ cao = sử dụng chỗ tối ưu = doanh thu tốt hơn
            """)
    
    with col3:
        st.markdown("#### 3. Điểm Hòa vốn")
        fig_bep = create_strategic_gauge(
            strategic['break_even_point'],
            "Breakeven Point",
            unit=" ₫",
            threshold_good=strategic['total_revenue'],
            threshold_bad=strategic['total_revenue'] * 1.5,
            is_inverse=True
        )
        st.plotly_chart(fig_bep, use_container_width=True)
        with st.expander("📖 Giải thích"):
            st.write(f"""
            **Điểm hòa vốn** là mức doanh thu cần đạt để không lãi không lỗ.
            
            - **Giá trị hiện tại:** {format_currency(strategic['break_even_point'])}
            - **Doanh thu hiện tại:** {format_currency(strategic['total_revenue'])}
            - **Ý nghĩa:** Đã vượt hòa vốn {format_percentage((strategic['total_revenue'] - strategic['break_even_point'])/strategic['break_even_point']*100) if strategic['break_even_point'] > 0 else 'N/A'}
            """)
    
    st.markdown("---")
    
    # ========== VÙNG 2: CẤU TRÚC CHI PHÍ ==========
    st.markdown("### 💰 Vùng 2: Cấu trúc Chi phí")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 4. Tỷ lệ Giá vốn/Giá bán")
        fig_cogs = create_strategic_gauge(
            strategic['cogs_ratio'],
            "COGS Ratio",
            unit="%",
            threshold_good=60,
            threshold_bad=80,
            is_inverse=True
        )
        st.plotly_chart(fig_cogs, use_container_width=True)
        st.caption(f"Chi phí: {format_currency(strategic['total_cost'])}")
    
    with col2:
        st.markdown("#### 5. Chi phí Nhân sự/DT")
        fig_personnel = create_strategic_gauge(
            strategic['personnel_cost_ratio'],
            "Personnel Cost %",
            unit="%",
            threshold_good=15,
            threshold_bad=25,
            is_inverse=True
        )
        st.plotly_chart(fig_personnel, use_container_width=True)
    
    with col3:
        st.markdown("#### 6. Chi phí Bán hàng/DT")
        fig_sales = create_strategic_gauge(
            strategic['sales_cost_ratio'],
            "Sales Cost %",
            unit="%",
            threshold_good=5,
            threshold_bad=10,
            is_inverse=True
        )
        st.plotly_chart(fig_sales, use_container_width=True)
    
    with col4:
        st.markdown("#### 7. Chi phí Vận hành/DT")
        fig_opex = create_strategic_gauge(
            strategic['operating_cost_ratio'],
            "Operating Cost %",
            unit="%",
            threshold_good=20,
            threshold_bad=35,
            is_inverse=True
        )
        st.plotly_chart(fig_opex, use_container_width=True)
        st.caption(f"OPEX: {format_currency(strategic['total_opex'])}")
    
    # Cost breakdown pie chart
    st.markdown("#### Phân tích Cấu trúc Chi phí")
    cost_data = pd.DataFrame({
        'Loại chi phí': ['Giá vốn (COGS)', 'Chi phí Nhân sự', 'Chi phí Bán hàng', 'Chi phí Vận hành khác'],
        'Giá trị': [
            strategic['total_cost'],
            strategic['total_opex'] * 0.5,  # Personnel = 50% OPEX
            strategic['total_revenue'] * strategic['sales_cost_ratio'] / 100,
            strategic['total_opex'] * 0.5   # Other operating costs
        ]
    })
    
    fig_cost_pie = px.pie(
        cost_data,
        values='Giá trị',
        names='Loại chi phí',
        title='Phân bổ Chi phí',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_cost_pie.update_traces(textinfo='label+percent', hovertemplate='<b>%{label}</b><br>%{value:,.0f} ₫<br>%{percent}<extra></extra>')
    fig_cost_pie.update_layout(height=400, margin=dict(t=40, b=10))
    st.plotly_chart(fig_cost_pie, use_container_width=True)
    
    st.markdown("---")
    
    # ========== VÙNG 3: KHẢ NĂNG SINH LỜI ==========
    st.markdown("### 📈 Vùng 3: Khả năng Sinh lời")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 8. Tỷ suất LN/Doanh thu")
        fig_npm = create_strategic_gauge(
            strategic['profit_margin_ratio'],
            "Net Profit Margin",
            unit="%",
            threshold_good=15,
            threshold_bad=5,
            is_inverse=False
        )
        st.plotly_chart(fig_npm, use_container_width=True)
        with st.expander("📊 Chi tiết"):
            st.write(f"""
            - **Lãi gộp:** {format_currency(strategic['total_gross_profit'])}
            - **OPEX:** {format_currency(strategic['total_opex'])}
            - **Lãi ròng:** {format_currency(strategic['net_profit'])}
            - **Doanh thu:** {format_currency(strategic['total_revenue'])}
            """)
    
    with col2:
        st.markdown("#### 9. ROI - Lợi nhuận/Vốn ĐT")
        fig_roi = create_strategic_gauge(
            strategic['roi'],
            "Return on Investment",
            unit="%",
            threshold_good=20,
            threshold_bad=10,
            is_inverse=False
        )
        st.plotly_chart(fig_roi, use_container_width=True)
        with st.expander("📖 Giải thích"):
            st.write(f"""
            **ROI** đo lường hiệu quả đầu tư. ROI cao = hiệu quả sử dụng vốn tốt.
            
            - **ROI hiện tại:** {strategic['roi']:.1f}%
            - **Chuẩn tốt:** ≥ 20%
            - **Công thức:** Lãi ròng / Tổng vốn đầu tư × 100
            """)
    
    with col3:
        st.markdown("#### 10. ROE - Lợi nhuận/Vốn CSH")
        fig_roe = create_strategic_gauge(
            strategic['roe'],
            "Return on Equity",
            unit="%",
            threshold_good=25,
            threshold_bad=15,
            is_inverse=False
        )
        st.plotly_chart(fig_roe, use_container_width=True)
        with st.expander("📖 Giải thích"):
            st.write(f"""
            **ROE** đo lường lợi nhuận tạo ra từ vốn chủ sở hữu.
            
            - **ROE hiện tại:** {strategic['roe']:.1f}%
            - **Chuẩn tốt:** ≥ 25%
            - **Công thức:** Lãi ròng / Vốn chủ sở hữu × 100
            """)
    
    # Profitability trend chart
    st.markdown("#### Xu hướng Sinh lời theo thời gian")
    
    # Group by month to show trend
    monthly_data = filtered_tours.copy()
    monthly_data['month'] = pd.to_datetime(monthly_data['booking_date']).dt.to_period('M')
    monthly_profit = monthly_data.groupby('month').agg({
        'revenue': 'sum',
        'gross_profit': 'sum',
        'opex': 'sum'
    }).reset_index()
    monthly_profit['net_profit'] = monthly_profit['gross_profit'] - monthly_profit['opex']
    monthly_profit['profit_margin'] = (monthly_profit['net_profit'] / monthly_profit['revenue'] * 100).fillna(0)
    monthly_profit['month_str'] = monthly_profit['month'].astype(str)
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=monthly_profit['month_str'],
        y=monthly_profit['net_profit'],
        name='Lãi ròng',
        marker_color='#00CC96',
        yaxis='y'
    ))
    fig_trend.add_trace(go.Scatter(
        x=monthly_profit['month_str'],
        y=monthly_profit['profit_margin'],
        name='Tỷ suất LN (%)',
        mode='lines+markers',
        marker_color='#EF553B',
        yaxis='y2'
    ))
    
    fig_trend.update_layout(
        height=400,
        xaxis_title="Tháng",
        yaxis=dict(title="Lãi ròng (₫)", side='left'),
        yaxis2=dict(title="Tỷ suất LN (%)", overlaying='y', side='right'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("---")
    
    # Summary insights
    st.markdown("### 💡 Tổng kết Chỉ số Chiến lược")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("#### ✅ Điểm mạnh")
        strengths = []
        if strategic['dso'] < 30:
            strengths.append(f"✓ DSO tốt ({strategic['dso']:.1f} ngày) - Thu tiền nhanh")
        if strategic['inventory_turnover'] >= 70:
            strengths.append(f"✓ Vòng quay tốt ({strategic['inventory_turnover']:.1f}%) - Sử dụng capacity hiệu quả")
        if strategic['profit_margin_ratio'] >= 15:
            strengths.append(f"✓ Tỷ suất LN cao ({strategic['profit_margin_ratio']:.1f}%) - Sinh lời tốt")
        if strategic['roi'] >= 20:
            strengths.append(f"✓ ROI xuất sắc ({strategic['roi']:.1f}%) - Đầu tư hiệu quả")
        if strategic['roe'] >= 25:
            strengths.append(f"✓ ROE mạnh ({strategic['roe']:.1f}%) - Vốn CSH sinh lời cao")
        
        if strengths:
            for s in strengths:
                st.write(s)
        else:
            st.write("Chưa có điểm mạnh nổi bật trong kỳ này")
    
    with col2:
        st.warning("#### ⚠️ Cần cải thiện")
        warnings = []
        if strategic['dso'] >= 60:
            warnings.append(f"⚠ DSO cao ({strategic['dso']:.1f} ngày) - Cần thu tiền nhanh hơn")
        if strategic['inventory_turnover'] < 50:
            warnings.append(f"⚠ Vòng quay thấp ({strategic['inventory_turnover']:.1f}%) - Tối ưu capacity")
        if strategic['cogs_ratio'] >= 80:
            warnings.append(f"⚠ Giá vốn cao ({strategic['cogs_ratio']:.1f}%) - Đàm phán lại giá")
        if strategic['operating_cost_ratio'] >= 35:
            warnings.append(f"⚠ Chi phí vận hành cao ({strategic['operating_cost_ratio']:.1f}%) - Cắt giảm OPEX")
        if strategic['profit_margin_ratio'] < 5:
            warnings.append(f"⚠ Tỷ suất LN thấp ({strategic['profit_margin_ratio']:.1f}%) - Cải thiện lợi nhuận")
        if strategic['roi'] < 10:
            warnings.append(f"⚠ ROI yếu ({strategic['roi']:.1f}%) - Xem xét lại đầu tư")
        
        if warnings:
            for w in warnings:
                st.write(w)
        else:
            st.write("Các chỉ số đều ở mức tốt!")

st.markdown("---")

# Footer
st.markdown("""
    <div style='text-align: center; padding: 20px; color: #666;'>
        <p>📊 Vietravel Business Intelligence Dashboard</p>
        <p>Cập nhật lần cuối: {}</p>
    </div>
""".format(datetime.now().strftime("%d/%m/%Y %H:%M")), unsafe_allow_html=True)
