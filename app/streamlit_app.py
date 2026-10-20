# streamlit run trafficflow_ai.py
import json
from pathlib import Path
import numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'app_data' if (ROOT / 'app_data' / 'meta.json').exists() else ROOT / 'app_data_mock'
COLORS = {0: '#34C759', 1: '#FF9F0A', 2: '#FF453A', -1: '#9AA5B1'}
NAMES = {0: 'Free-flowing', 1: 'Slow', 2: 'Congested', -1: 'No data'}

LABELS = {'trafficflow_st': 'TrafficFlow-ST (road graph)',
          'no_graph': 'Same model, no road graph',
          'historical_average': 'Historical Average',
          'persistence': 'Persistence (last value)',
          'trafficflow_blend': 'TrafficFlow-ST + historical pattern'}

def label(m):
    return LABELS.get(m, m)

@st.cache_data
def load(path):
    p = Path(path)
    meta = json.loads((p / 'meta.json').read_text())
    sensors = pd.read_csv(p / 'sensors.csv')
    edges = pd.read_csv(p / 'edges.csv')
    ts = pd.to_datetime(pd.read_csv(p / 'timestamps.csv')['timestamp'])
    actual = np.load(p / 'actual.npy').astype(np.float32)
    preds = {m: np.load(p / f'pred_{m}.npy').astype(np.float32)
             for m in meta['models'] if (p / f'pred_{m}.npy').exists()}
    metrics = json.loads((p / 'metrics.json').read_text())
    return meta, sensors, edges, ts, actual, preds, metrics

def levels(speed, ff, th):
    r = speed / np.array(ff)
    out = np.where(r >= th['free'], 0, np.where(r >= th['slow'], 1, 2))
    return np.where(np.isnan(speed), -1, out)

def network_map(sensors, edges, lv, title, dark=False, lock=False):
    fig = go.Figure()
    xs, ys = [], []
    for s, t in zip(edges.source, edges.target):
        xs += [sensors.longitude[s], sensors.longitude[t], None]
        ys += [sensors.latitude[s], sensors.latitude[t], None]
    fig.add_trace(go.Scattermap(lon=xs, lat=ys, mode='lines',
                                line=dict(width=1, color='#6B7280' if dark else '#94A3B8'),
                                hoverinfo='skip', showlegend=False))
    for code in (0, 1, 2, -1):
        m = lv == code
        fig.add_trace(go.Scattermap(lon=sensors.longitude[m], lat=sensors.latitude[m],
                                    mode='markers', marker=dict(size=9, color=COLORS[code]),
                                    name=NAMES[code],
                                    text='Sensor ' + sensors.sensor_id[m].astype(str)))
    fig.update_layout(
        map_style='carto-darkmatter' if dark else 'carto-voyager',
        title=title, height=450,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E6E9EF' if dark else '#334155'),
        legend=dict(orientation='h', yanchor='bottom', y=0.01, xanchor='left', x=0.01,
                    bgcolor='rgba(26,29,36,0.75)' if dark else 'rgba(244,246,249,0.85)'),
        map=dict(center=dict(lat=sensors.latitude.mean(), lon=sensors.longitude.mean()), zoom=9.5),
        uirevision='locked' if lock else 'free')
    if lock:
        fig.update_layout(dragmode=False)
    return fig

st.set_page_config(page_title='TrafficFlow AI', page_icon='🚦', layout='wide')
if not (DATA / 'meta.json').exists():
    st.error('No data found. Run python mock/make_mock_data.py, or wait for app_data/ from the backend.')
    st.stop()
meta, sensors, edges, ts, actual, preds, metrics = load(str(DATA))
if not preds:
    st.error('No prediction files (pred_<model>.npy) found in the data folder.')
    st.stop()
T = len(ts)
th = meta['thresholds']
dark = getattr(st.context.theme, 'type', 'light') == 'dark'

def default_index(ts, T):
    if T - 13 >= 90:
        return 90
    # first weekday 08:00 in the window, so the demo opens on rush hour
    for i in range(T - 12):
        x = ts.iloc[i]
        if x.hour == 8 and x.minute == 0 and x.weekday() < 5:
            return i
    return 0

st.sidebar.title('TrafficFlow AI')
names = list(preds)
model = st.sidebar.selectbox('Model', names,
        index=names.index(meta['main_model']) if meta['main_model'] in names else 0,
        format_func=label)
t = st.sidebar.slider('Forecast issued at (index)', 0, T - 13, default_index(ts, T))
st.sidebar.caption(str(ts.iloc[t]))
minutes = st.sidebar.select_slider('Minutes ahead', options=list(range(5, 65, 5)), value=30)
map_mode = st.sidebar.radio('Map style', ['Light (Google-style)', 'Dark', 'Match app theme'], index=0)
lock_maps = st.sidebar.checkbox('Lock map view (both maps stay aligned)', value=True)
map_dark = dark if map_mode == 'Match app theme' else (map_mode == 'Dark')
h = minutes // 5 - 1

pred_now = preds[model][t, h, :]
actual_now = actual[t + h + 1, :]
lv_pred = levels(pred_now, meta['free_flow_speed'], th)
lv_act = levels(actual_now, meta['free_flow_speed'], th)
ok = ~np.isnan(actual_now)

st.markdown('## TrafficFlow AI')
st.caption(f"Team BitZen · {meta['dataset']} · {meta['region']}")

tab_map, tab_sensor, tab_models, tab_rob, tab_about = st.tabs(
    ['Forecast Map', 'Sensor Detail', 'Model Comparison', 'Robustness & Ablation', 'About'])

with tab_map:
    st.subheader('Forecast Map')
    st.caption(f"Forecast issued {ts.iloc[t]:%a %d %b %Y, %H:%M} · showing {ts.iloc[t + h + 1]:%H:%M} (+{minutes} min)")
    mae = float(np.nanmean(np.abs(pred_now - actual_now))) if ok.any() else float('nan')
    match = float((lv_pred[ok] == lv_act[ok]).mean() * 100) if ok.any() else float('nan')
    c1, c2 = st.columns(2)
    c1.metric('Error at this moment (mph)', f'{mae:.2f}',
              help='Average |predicted - actual| over all sensors at this one forecast time and horizon. A single moment is noisy; the full-test average is on the Model Comparison tab.')
    c2.metric('Congestion level matched at this moment', f'{match:.0f}%',
              help='Average |predicted - actual| over all sensors at this one forecast time and horizon. A single moment is noisy; the full-test average is on the Model Comparison tab.')
    st.caption('These two numbers describe one moment only. The real average over the whole test period is on the Model Comparison tab.')
    m1, m2 = st.columns(2)
    m1.plotly_chart(network_map(sensors, edges, lv_pred, f'Predicted (+{minutes} min)', dark=map_dark, lock=lock_maps), width='stretch', key='map_pred', config={'scrollZoom': not lock_maps})
    m2.plotly_chart(network_map(sensors, edges, lv_act, f'Actual (+{minutes} min)', dark=map_dark, lock=lock_maps), width='stretch', key='map_act', config={'scrollZoom': not lock_maps})

with tab_sensor:
    st.subheader('Sensor Detail')
    ff = np.array(meta['free_flow_speed'])
    ids = sensors['sensor_id'].tolist()
    default_pos = int(np.nanargmin(actual_now / ff)) if ok.any() else 0
    sid = st.selectbox('Sensor', ids, index=default_pos, key='sensor_pick',
                       format_func=lambda i: f'Sensor {i}')
    others = [m for m in names if m != model]
    overlay = st.multiselect('Overlay other models', others, default=[], key='overlay_models',
                             format_func=label)

    t0 = max(0, t - 11)
    x_hist = ts.iloc[t0:t + 1]
    x_fut = ts.iloc[t + 1:t + 13]
    y_hist = actual[t0:t + 1, sid]
    y_fut = actual[t + 1:t + 13, sid]
    y_pred = preds[model][t, :, sid]

    ink = '#E6E9EF' if dark else '#1F2A44'
    blue = '#5AA0F0' if dark else '#4A90E2'
    grey = '#9AA5B1'
    extra = ['#A78BFA', '#38BDF8', '#F472B6', '#94A3B8']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_hist, y=y_hist, mode='lines', name='Actual (past 60 min)',
                             line=dict(color=ink, width=2)))
    fig.add_trace(go.Scatter(x=x_fut, y=y_fut, mode='lines', name='Actual (future)',
                             line=dict(color=ink, width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=x_fut, y=y_pred, mode='lines+markers', name=f'Forecast ({label(model)})',
                             line=dict(color=blue, width=2, dash='dash'),
                             fill='tonexty', fillcolor='rgba(74,144,226,0.20)'))
    for i, m in enumerate(overlay):
        fig.add_trace(go.Scatter(x=x_fut, y=preds[m][t, :, sid], mode='lines', name=f'Forecast ({label(m)})',
                                 line=dict(color=extra[i % len(extra)], width=1.5, dash='dash')))

    xt = ts.iloc[t].to_pydatetime()
    fig.add_shape(type='line', x0=xt, x1=xt, y0=0, y1=1, yref='paper',
                  line=dict(color=grey, width=1.5, dash='dash'))
    fig.add_annotation(x=xt, y=1, yref='paper', text='Forecast time', showarrow=False,
                       yanchor='bottom', font=dict(color=grey))
    fig.add_hline(y=float(ff[sid]), line_dash='dot', line_color=grey,
                  annotation_text='Free-flow speed', annotation_position='bottom right')

    fig.update_layout(title=f'Sensor {sid}: past speed, forecast and actual',
                      xaxis_title='Time', yaxis_title='Speed (mph)', height=480,
                      template='plotly_dark' if dark else 'plotly_white',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation='h', y=-0.2),
                      margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(rangemode='tozero')
    st.plotly_chart(fig, width='stretch')

    valid = ~np.isnan(y_fut)
    if valid.any():
        s_mae = float(np.mean(np.abs(y_pred[valid] - y_fut[valid])))
        st.caption(f'Sensor {sid} · mean absolute error over the next 60 min: {s_mae:.2f} mph · '
                   f'free-flow speed {ff[sid]:.0f} mph')
    else:
        st.caption('No observed values for this sensor in the next 60 minutes.')
with tab_models:
    st.subheader('Model Comparison')
    mnames = [m for m in meta['models'] if m in metrics]
    horizons = ['15', '30', '60']
    groups = [('mae', 'MAE (mph)', False), ('rmse', 'RMSE (mph)', False),
              ('mape', 'MAPE (%)', False), ('accuracy', 'Congestion accuracy', True),
              ('macro_f1', 'Macro-F1', True)]

    cols, higher = [], {}
    for key, lbl, hib in groups:
        for hz in horizons:
            c = f'{lbl} @{hz}min'
            cols.append((c, key, hz))
            higher[c] = hib

    rows = {}
    for m in mnames:
        row = {}
        for c, key, hz in cols:
            try:
                row[c] = float(metrics[m][key][hz])
            except (KeyError, TypeError, ValueError):
                row[c] = np.nan
        rows[m] = row
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index = [label(m) for m in df.index]
    df.index.name = 'Model'

    hl_bg = 'rgba(90,160,240,0.30)' if dark else 'rgba(74,144,226,0.20)'

    def highlight(col):
        if col.isna().all():
            return [''] * len(col)
        best = col.max() if higher[col.name] else col.min()
        return [f'background-color: {hl_bg}; font-weight: 700' if v == best else '' for v in col]

    fmt = {c: ('{:.1%}' if key == 'accuracy' else '{:.3f}' if key == 'macro_f1' else '{:.2f}')
           for c, key, hz in cols}
    st.dataframe(df.style.apply(highlight, axis=0).format(fmt, na_rep='n/a'), width='stretch')
    st.caption('Best value per column is highlighted: lowest for MAE, RMSE and MAPE; '
               'highest for accuracy and macro-F1. Source: metrics.json.')

    blue_shades = {}
    palette = ['#4A90E2', '#A78BFA', '#38BDF8', '#F472B6', '#94A3B8', '#FBBF24']
    for i, m in enumerate(mnames):
        blue_shades[m] = palette[i % len(palette)]
    main_color = '#5AA0F0' if dark else '#4A90E2'
    if meta['main_model'] in blue_shades:
        blue_shades[meta['main_model']] = main_color

    figb = go.Figure()
    for m in mnames:
        vals = []
        for hz in horizons:
            try:
                vals.append(float(metrics[m]['mae'][hz]))
            except (KeyError, TypeError, ValueError):
                vals.append(None)
        figb.add_trace(go.Bar(x=[f'{hz} min' for hz in horizons], y=vals, name=label(m),
                              marker_color=blue_shades[m]))
    figb.update_layout(barmode='group', title='MAE by forecast horizon',
                       xaxis_title='Forecast horizon (minutes ahead)', yaxis_title='MAE (mph)',
                       height=440, template='plotly_dark' if dark else 'plotly_white',
                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       legend=dict(orientation='h', y=-0.25),
                       margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(figb, width='stretch')
with tab_rob:
    st.subheader('Robustness & Ablation')
    rob_path = DATA / 'robustness.csv'
    abl_path = DATA / 'ablation.csv'
    plot_tpl = 'plotly_dark' if dark else 'plotly_white'
    main_color = '#5AA0F0' if dark else '#4A90E2'
    other_colors = ['#A78BFA', '#38BDF8', '#F472B6', '#94A3B8', '#FBBF24']

    if rob_path.exists():
        rob = pd.read_csv(rob_path)
        figr = go.Figure()
        k = 0
        for m in rob['model'].unique():
            d = rob[rob['model'] == m].sort_values('missing_pct')
            if m == meta['main_model']:
                col, w = main_color, 3
            else:
                col, w = other_colors[k % len(other_colors)], 2
                k += 1
            figr.add_trace(go.Scatter(x=d['missing_pct'], y=d['mae'], mode='lines+markers',
                                      name=label(m), line=dict(color=col, width=w)))
        figr.update_layout(title='Forecast error vs. share of missing sensors',
                           xaxis_title='Missing sensors (%)', yaxis_title='MAE (mph)',
                           height=420, template=plot_tpl,
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           legend=dict(orientation='h', y=-0.25),
                           margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(figr, width='stretch')
        st.caption('Lower is better. A flatter line means the model copes better when sensors drop out. '
                   'Source: robustness.csv.')
    else:
        st.info('robustness.csv not found in the data folder.')

    st.markdown('#### Ablation study')
    if abl_path.exists():
        abl = pd.read_csv(abl_path).set_index('variant')
        abl = abl.rename(columns={'mae_15': 'MAE @15 min (mph)', 'mae_30': 'MAE @30 min (mph)',
                                  'mae_60': 'MAE @60 min (mph)'})
        hl_bg = 'rgba(90,160,240,0.30)' if dark else 'rgba(74,144,226,0.20)'

        def best_low(col):
            b = col.min()
            return [f'background-color: {hl_bg}; font-weight: 700' if v == b else '' for v in col]

        st.dataframe(abl.style.apply(best_low, axis=0).format('{:.2f}'), width='stretch')
        st.caption('Each row removes one component from the full model. Lowest MAE per column is highlighted. '
                   'Source: ablation.csv.')
    else:
        st.info('ablation.csv not found in the data folder.')

with tab_about:
    st.subheader('About TrafficFlow AI')
    year = ts.iloc[0].year
    arch = ROOT / 'docs' / 'architecture.png'

    st.markdown('**Problem**')
    st.write('Congestion forms and spreads across connected roads within minutes, '
             'but most tools only show current conditions.')
    st.markdown('**Solution**')
    st.write('TrafficFlow AI forecasts traffic speed for every road segment up to 60 minutes ahead '
             'using a graph-based spatiotemporal model, and turns forecasts into congestion levels '
             'shown on an interactive map.')

    st.markdown('**Model**')
    if 'trafficflow_blend' in meta['models']:
        st.write('The final model is a blend of a neural network (a graph-based spatiotemporal model) '
                 'and the historical pattern for each sensor and time of day.')
    else:
        st.write('The main model is a graph-based spatiotemporal neural network.')

    st.markdown('**Architecture**')
    if arch.exists():
        st.image(str(arch), width='stretch')
    else:
        st.info('Architecture diagram will appear here once docs/architecture.png is added.')

    st.markdown('**Data**')
    st.write(f"{meta['dataset']}, {meta['n_sensors']} loop-detector sensors in {meta['region']}, "
             f"{meta['step_minutes']}-minute readings, {year}. Only aggregated public sensor speeds are used. "
             'No personal or identifiable mobility data.')

    st.markdown('**Limitations**')
    blend_prefix = ('The final model blends a neural network with the historical pattern, so on unusual days '
                    '(incidents, holidays) it leans partly on typical behaviour. '
                    if 'trafficflow_blend' in meta['models'] else '')
    st.write(f"{blend_prefix}Trained on {meta['region']} highways, not city streets worldwide. "
             f"The data is from {year}, and traffic patterns have changed. "
             'Forecast error grows with the horizon, so long-range forecasts are less reliable. '
             'Sensors cover mainly major highways. '
             'Forecasts are decision support, not a replacement for human traffic control.')

    st.caption('Team BitZen · AI Innovation Hackathon 2026')
