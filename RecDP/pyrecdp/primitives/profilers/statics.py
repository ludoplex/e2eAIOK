from pyrecdp.core.schema import SeriesSchema
from pyrecdp.core.utils import is_text_series
from pyrecdp.core.utils import Timer
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyrecdp.core.dataframe import DataFrameAPI
import matplotlib.pyplot as plt

def draw_xy_scatter_plot(xy_scatter_features, feature_data, y, row_height, n_plot_per_row):
    import math

    n_feat = len(xy_scatter_features)
    n_row = math.ceil(n_feat / n_plot_per_row)
    n_col = min(n_feat, n_plot_per_row)
    label = y.name

    height = int(n_row * 3)
    fig, axs = plt.subplots(nrows=n_row, ncols=n_col, figsize=(10,height))

    X = DataFrameAPI().instiate(feature_data)
    sampled_data = X.may_sample(nrows = 1000)
    y = sampled_data[y.name]

    for idx, c_name in tqdm(enumerate(xy_scatter_features), total=len(xy_scatter_features)):
        feature = sampled_data[c_name]
        # if string, wrap when too long
        sch = SeriesSchema(feature)
        is_feature_string = bool((sch.is_string or sch.is_categorical_and_string))

        row_id = int(idx / n_plot_per_row)
        col_id = idx % n_plot_per_row
        if is_feature_string:
            tmp = feature.str.slice(0, 12, 1)
            axs[row_id, col_id].scatter(x=tmp, y=y, s=5)
        else:
            axs[row_id, col_id].scatter(x=feature, y=y, s=5)

        axs[row_id, col_id].set_xlabel(c_name)
        axs[row_id, col_id].set_ylabel(label)

    print("prepare xy scatter plot completed")

    fig.tight_layout()
    return fig

def draw_mapbox_plot(mapbox_scatter_features, feature_data):
    import plotly.graph_objs as go

    from shapely.geometry import MultiPoint
    fig_list = go.Figure()

    for c_name in mapbox_scatter_features:
        feature = feature_data[c_name]
        if len(feature) > 10000:
            feature = feature.sample(n=10000, random_state=123)
        coords = feature.to_list()
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        fig_list.add_trace(
            go.Scattermapbox(
                lat=lats,
                lon=lons,
                name=c_name
            )
        )
        points = MultiPoint(coords)
    fig_list.update_layout(
        autosize=False,
        mapbox=dict(
            style='open-street-map',
            bearing=0,
            center=dict(
                lat=points.centroid.x,
                lon=points.centroid.y
            ),
            pitch=0,
            zoom=8
        ),
    )
        
    return fig_list
class StatisticsFeatureGenerator():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_prepare(self, pipeline, children, max_idx):
        return pipeline, children[0], max_idx
    
    def update_feature_statistics(self, X, y):
        overview_detail = {}
        overview_info = {
            'Number of Features': X.shape[1],
            'Number of Rows': X.shape[0],
        }
        length = X.shape[0]
        for feature_name in X.columns:
            with Timer(f"prepare info for {feature_name}"):
                feature = X[feature_name]
                desc_info = {
                    k: v
                    for k, v in feature.describe().to_dict().items()
                    if k not in ['count']
                }
                n_unique = desc_info['unique'] if 'unique' in desc_info else feature.nunique()
                feature_type = SeriesSchema(feature).dtype_str
                feature_type = "text" if is_text_series(feature) else feature_type

                stat = {'type': feature_type, 'unique': {"u": n_unique, "m": length}, 'quantile':desc_info}
                if feature_name not in overview_detail:
                    overview_detail[feature_name] = stat

        data_stats = {"overview": (overview_info, overview_detail)}
        if y is not None:
            interactions_detail = self.get_interactive_plot(X, y)
        data_stats['interactions']=(dict(), "")

        return data_stats
    
    def get_interactive_plot(self, feature_data, y):
        import base64
        from io import BytesIO
        # we will create types of plot
        xy_scatter_features = []
        mapbox_scatter_features = []
        word_cloud_features = []
        time_series_features = []

        for c_name in feature_data.columns:
            feature = feature_data[c_name]
            # if string, wrap when too long
            sch = SeriesSchema(feature)
            is_coord = bool(sch.is_coordinates)

            if is_coord:
                mapbox_scatter_features.append(c_name)
            else:
                xy_scatter_features.append(c_name)

        ret = {}
        ret = {"error": False}

        # draw xy scatter
        if xy_scatter_features:
            row_height = 300
            n_plot_per_row = 2

            with Timer("Draw xy scatter plot"):
                fig = draw_xy_scatter_plot(xy_scatter_features, feature_data, y, row_height, n_plot_per_row)
                tmpfile = BytesIO()
                fig.savefig(tmpfile, format='png')
                plt.close(fig)
                encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
                ret['html'] = f"<div><img src=\'data:image/png;base64,{encoded}\'></div>"

        # draw mapbox
        if mapbox_scatter_features:
            with Timer("Draw mapbox plot"):
                fig_list = draw_mapbox_plot(mapbox_scatter_features, feature_data)
            from plotly.offline import plot
            ret['html'] += plot(fig_list, output_type='div')

        return ret
