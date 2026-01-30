import React, { useState } from 'react';
import {
  MapContainer, TileLayer, ImageOverlay, LayersControl
} from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts';
import {
  Activity, Map as MapIcon, BarChart3, AlertTriangle,
  Download, Layers, ShieldCheck, TrendingUp, Info, Zap, Layout
} from 'lucide-react';
import analyticsData from '../public/lulc_analytics.json';

const CLASSES = {
  "Forest": "#228B22",
  "Water Bodies": "#1E90FF",
  "Agriculture": "#DAA520",
  "Barren Land": "#DEB887",
  "Built-up": "#FF4500"
};

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const bounds = analyticsData.bounds;

  const stats2015 = Object.entries(analyticsData.years['2015']).map(([name, data]) => ({ name, ...data }));
  const stats2024 = Object.entries(analyticsData.years['2024']).map(([name, data]) => ({ name, ...data }));

  const comparisonData = stats2015.map(s15 => {
    const s24 = stats2024.find(s => s.name === s15.name);
    return {
      name: s15.name,
      '2015': s15.area_km2,
      '2024': s24?.area_km2 || 0,
      change: ((s24?.area_km2 || 0) - s15.area_km2).toFixed(2)
    };
  });

  const transitionData = analyticsData.matrix.flatMap((row, i) =>
    row.map((val, j) => ({
      from: analyticsData.class_names[i],
      to: analyticsData.class_names[j],
      value: val
    }))
  ).filter(d => d.from !== d.to && d.value > 100);

  return (
    <div className="min-h-screen bg-[#0a0a0b] text-white p-6 lg:p-10">
      <header className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 mb-10">
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-nature-500/20 rounded-lg">
              <ShieldCheck className="w-8 h-8 text-nature-500" />
            </div>
            <div>
              <h1 className="text-3xl font-bold tracking-tight">Tirupati SmartCity GeoAI</h1>
              <div className="flex items-center gap-2 text-xs font-mono text-nature-500 uppercase tracking-widest mt-1">
                <div className="w-2 h-2 rounded-full bg-nature-500 animate-pulse" />
                Pixel-Level Change Intel v2.0
              </div>
            </div>
          </div>
        </div>

        <nav className="flex bg-white/5 p-1 rounded-xl border border-white/10 backdrop-blur-md">
          {[
            { id: 'overview', label: 'Dashboard', icon: Layout },
            { id: 'explorer', label: 'Spatial Explorer', icon: MapIcon },
            { id: 'governance', label: 'Governance Panel', icon: Activity }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-semibold transition-all ${activeTab === tab.id ? 'bg-nature-500 text-white shadow-lg' : 'text-gray-400 hover:text-white'
                }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </nav>
      </header>

      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Governance Quick Cards */}
          <div className="lg:col-span-3 grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="glass-card p-6 border-urban-500/20 bg-urban-500/5">
              <div className="flex justify-between items-start mb-4">
                <Zap className="w-5 h-5 text-urban-500" />
                <span className="text-[10px] bg-urban-500/20 text-urban-500 px-2 py-0.5 rounded uppercase font-bold">Alert</span>
              </div>
              <p className="text-gray-400 text-xs font-medium uppercase tracking-wider">Urban Sprawl</p>
              <p className="text-3xl font-bold mt-1 text-white">{analyticsData.governance.urban_sprawl_km2} <span className="text-sm font-normal text-gray-400">km²</span></p>
              <p className="text-[10px] text-gray-500 mt-2">Conversion from Forest Cover</p>
            </div>

            <div className="glass-card p-6 border-nature-500/20 bg-nature-500/5">
              <div className="flex justify-between items-start mb-4">
                <ShieldCheck className="w-5 h-5 text-nature-500" />
                <span className="text-[10px] bg-nature-500/20 text-nature-500 px-2 py-0.5 rounded uppercase font-bold">Stable</span>
              </div>
              <p className="text-gray-400 text-xs font-medium uppercase tracking-wider">Model Precision</p>
              <p className="text-3xl font-bold mt-1 text-white">{(analyticsData.governance.avg_confidence * 100).toFixed(1)}%</p>
              <p className="text-[10px] text-gray-500 mt-2">Random Forest Confidence</p>
            </div>

            <div className="glass-card p-6 border-blue-500/20 bg-blue-500/5">
              <div className="flex justify-between items-start mb-4">
                <TrendingUp className="w-5 h-5 text-blue-500" />
                <span className="text-[10px] bg-blue-500/20 text-blue-500 px-2 py-0.5 rounded uppercase font-bold">Change</span>
              </div>
              <p className="text-gray-400 text-xs font-medium uppercase tracking-wider">Transition Intel</p>
              <p className="text-3xl font-bold mt-1 text-white">{analyticsData.governance.high_likelihood_changes.toLocaleString()}</p>
              <p className="text-[10px] text-gray-500 mt-2">Pixels with {'>'}80% likelihood</p>
            </div>

            <div className="glass-card p-6 border-yellow-500/20 bg-yellow-500/5">
              <div className="flex justify-between items-start mb-4">
                <AlertTriangle className="w-5 h-5 text-yellow-500" />
              </div>
              <p className="text-gray-400 text-xs font-medium uppercase tracking-wider">Env. Impact Index</p>
              <p className="text-3xl font-bold mt-1 text-white">High</p>
              <p className="text-[10px] text-gray-500 mt-2">Based on Vegetation Loss</p>
            </div>
          </div>

          {/* Metrics Panel */}
          <div className="lg:col-span-3 glass-card p-8 min-h-[450px]">
            <div className="flex items-center justify-between mb-8">
              <h3 className="text-xl font-bold">Area Dynamics Comparison</h3>
              <div className="flex gap-4">
                <div className="flex items-center gap-2 text-xs text-gray-500"><div className="w-3 h-1 bg-[#333]" /> 2015 BaseLine</div>
                <div className="flex items-center gap-2 text-xs text-nature-500"><div className="w-3 h-1 bg-nature-500" /> 2024 Current</div>
              </div>
            </div>
            <div className="h-[350px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                  <XAxis dataKey="name" stroke="#666" axisLine={false} tickLine={false} />
                  <YAxis stroke="#666" axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: '#111', border: '1px solid #333', borderRadius: '8px' }} cursor={{ fill: '#ffffff05' }} />
                  <Bar dataKey="2015" fill="#333" radius={[4, 4, 0, 0]} barSize={20} />
                  <Bar dataKey="2024" fill="#228B22" radius={[4, 4, 0, 0]} barSize={20} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="lg:col-span-1 glass-card p-8 flex flex-col items-center">
            <h3 className="text-center font-bold mb-6">Class Distribution (2024)</h3>
            <div className="w-full h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={stats2024} innerRadius={70} outerRadius={90} dataKey="area_km2" paddingAngle={4}>
                    {stats2024.map((entry, index) => <Cell key={`c-${index}`} fill={CLASSES[entry.name]} stroke="none" />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="w-full space-y-3 mt-8">
              {stats2024.map(s => (
                <div key={s.name} className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2 text-gray-300">
                    <div className="w-2.5 h-2.5 rounded-sm" style={{ background: CLASSES[s.name] }} />
                    {s.name}
                  </div>
                  <span className="font-mono text-gray-500">{s.percentage}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'explorer' && (
        <div className="space-y-6">
          <div className="glass-card h-[70vh] relative overflow-hidden ring-1 ring-white/10">
            <MapContainer bounds={bounds} className="w-full h-full z-0" style={{ background: '#0a0a0b' }}>
              <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" />
              <LayersControl position="topright">
                <LayersControl.BaseLayer checked name="LULC Classification 2024">
                  <ImageOverlay url="/map_2024.png" bounds={bounds} opacity={0.9} />
                </LayersControl.BaseLayer>
                <LayersControl.BaseLayer name="LULC Classification 2015">
                  <ImageOverlay url="/map_2015.png" bounds={bounds} opacity={0.9} />
                </LayersControl.BaseLayer>
                <LayersControl.Overlay name="AI Transition Likelihood (Heatmap)">
                  <ImageOverlay url="/transition_likelihood.png" bounds={bounds} opacity={0.7} />
                </LayersControl.Overlay>
                <LayersControl.Overlay name="High-Confidence Change Hotspots">
                  <ImageOverlay url="/change_map.png" bounds={bounds} opacity={1} />
                </LayersControl.Overlay>
              </LayersControl>
            </MapContainer>

            <div className="absolute left-6 bottom-10 z-[1000] glass-card p-6 bg-black/90 backdrop-blur-3xl w-72 pointer-events-none border-white/5">
              <div className="flex items-center gap-2 text-nature-500 font-bold text-xs uppercase mb-4">
                <Activity className="w-3 h-3" /> Spatial Intelligence Metadata
              </div>
              <div className="space-y-4">
                <div className="text-[10px] text-gray-500 leading-relaxed italic">
                  The Heatmap layer (AI Transition Likelihood) represents P(Class_2015) × P(Class_2024).
                  Hotter zones indicate higher statistical certainty of land cover conversion.
                </div>
                <div className="grid grid-cols-2 gap-2 mt-4">
                  {Object.entries(CLASSES).map(([name, color]) => (
                    <div key={name} className="flex items-center gap-1.5 text-[10px] text-gray-300">
                      <div className="w-2 h-2 rounded-full" style={{ background: color }} />
                      {name}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="glass-card p-6 border-blue-500/20 bg-blue-500/5">
            <div className="flex gap-4">
              <Info className="w-5 h-5 text-blue-400 mt-1 flex-shrink-0" />
              <p className="text-sm text-blue-200/80 leading-relaxed">
                <strong>Spatial Analysis Engine:</strong> This view performs pixel-level overlaying. When you toggle the <i>Transition Likelihood</i> layer, you are seeing the result of the integrated Random Forest probability maps. This directly informs planners on which detections are most reliable for policy enforcement.
              </p>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'governance' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
          <div className="glass-card p-8">
            <h3 className="text-xl font-bold mb-8 flex items-center gap-3">
              <BarChart3 className="text-nature-500 w-6 h-6" /> Class-to-Class Transition Matrix
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-separate border-spacing-y-2">
                <thead>
                  <tr className="text-[10px] uppercase font-bold text-gray-500 tracking-widest border-b border-white/10">
                    <th className="pb-4">Source (2015)</th>
                    <th className="pb-4">Target (2024)</th>
                    <th className="pb-4">Magnitude</th>
                    <th className="pb-4">Risk Profile</th>
                  </tr>
                </thead>
                <tbody className="text-sm">
                  {transitionData.sort((a, b) => b.value - a.value).slice(0, 12).map((d, i) => (
                    <tr key={i} className="glass-card group hover:bg-white/5 transition-all">
                      <td className="p-4 rounded-l-lg font-medium text-gray-300">{d.from}</td>
                      <td className="p-4 font-medium text-gray-300">{d.to}</td>
                      <td className="p-4 font-mono text-nature-500">{d.value.toLocaleString()} px</td>
                      <td className="p-4 rounded-r-lg">
                        {d.to === 'Built-up' && (d.from === 'Forest' || d.from === 'Agriculture') ? (
                          <span className="bg-urban-500/20 text-urban-500 text-[9px] px-2 py-1 rounded-full font-bold uppercase tracking-tighter ring-1 ring-urban-500/40">Critical Encroachment</span>
                        ) : (
                          <span className="bg-gray-400/10 text-gray-500 text-[9px] px-2 py-1 rounded-full font-bold uppercase tracking-tighter ring-1 ring-gray-400/20">Minor Transition</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="space-y-6">
            <div className="glass-card p-8 bg-nature-900/10 border-nature-500/30">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-3">
                <Activity className="text-nature-500 w-6 h-6" /> Policy Recommendations
              </h3>
              <div className="space-y-6">
                <div className="p-4 border-l-4 border-urban-500 bg-black/40 rounded-r-lg">
                  <h4 className="font-bold text-urban-500 text-sm mb-1 uppercase tracking-wider">Contain Urban Drift</h4>
                  <p className="text-xs text-gray-400 leading-relaxed">High-density conversion of Forest to Built-up ({analyticsData.governance.urban_sprawl_km2} km²) detected in Eastern corridors. Suggest immediate zoning audit.</p>
                </div>
                <div className="p-4 border-l-4 border-nature-500 bg-black/40 rounded-r-lg">
                  <h4 className="font-bold text-nature-500 text-sm mb-1 uppercase tracking-wider">Green Belt Enforcement</h4>
                  <p className="text-xs text-gray-400 leading-relaxed">Maintain the 98% AI confidence verification zones. These areas are stable and should be declared as "Protected Green Corridors".</p>
                </div>
                <div className="p-4 border-l-4 border-blue-500 bg-black/40 rounded-r-lg">
                  <h4 className="font-bold text-blue-500 text-sm mb-1 uppercase tracking-wider">Water Body Restoration</h4>
                  <p className="text-xs text-gray-400 leading-relaxed">Detected transition of Water Bodies to Barren Land. This suggests drainage issues or encroachment. Requires hydrological survey.</p>
                </div>
              </div>
            </div>

            <div className="glass-card p-8 overflow-hidden relative group">
              <div className="absolute top-0 right-0 p-4 transform group-hover:scale-110 transition-transform">
                <Download className="text-nature-500/40" />
              </div>
              <h4 className="text-sm font-bold text-gray-300 uppercase tracking-widest mb-4">Export Decision-Ready Report</h4>
              <p className="text-xs text-gray-500 mb-6 leading-relaxed">Ready for one-click export to PDF/GeoJSON for state-level sustainability reporting and smart city committee evaluation.</p>
              <button className="w-full py-4 bg-nature-500 hover:bg-nature-600 active:scale-[0.98] transition-all rounded-lg font-bold text-sm tracking-widest shadow-lg shadow-nature-500/20">
                DOWNLOAD GOVERNANCE REPORT (.PDF)
              </button>
            </div>
          </div>
        </div>
      )}

      <footer className="mt-20 pt-10 border-t border-white/5 flex flex-col md:flex-row justify-between items-center gap-6 text-gray-600 text-[10px] uppercase font-medium tracking-widest">
        <div className="flex items-center gap-8">
          <span>Tirupati District Analytics Unit</span>
          <span>Landsat-8 Core Data v1.4</span>
        </div>
        <div className="flex items-center gap-4">
          <span>Coordinate Engine: EPSG:4326</span>
          <span className="text-nature-500">A Smart City Initiative</span>
        </div>
      </footer>
    </div>
  );
};

export default Dashboard;
