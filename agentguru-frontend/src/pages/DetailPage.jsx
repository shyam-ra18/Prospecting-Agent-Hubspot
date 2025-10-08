import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import SignalCard from '../components/SignalCard';

const DetailPage = () => {
    const { researchId } = useParams();
    const [data, setData] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchSignals = async () => {
            setIsLoading(true);
            setError('');
            try {
                const url = `http://127.0.0.1:8000/signals/${researchId}`;
                const response = await fetch(url);
                const json = await response.json();

                if (response.ok) {
                    setData(json);
                } else {
                    setError(json.detail || "Failed to fetch signal data.");
                }
            } catch (err) {
                setError("Network error. Could not connect to signal endpoint.");
            } finally {
                setIsLoading(false);
            }
        };
        fetchSignals();
    }, [researchId]);

    if (isLoading) return <div className="p-8 text-center text-xl">Loading signals and score... ⏳</div>;
    if (error) return <div className="p-8 text-center text-red-600 text-xl">Error: {error}</div>;
    if (!data) return <div className="p-8 text-center text-xl">No data found.</div>;

    // --- Helper Functions for UI ---
    const getScoreColor = (score) => {
        if (score >= 70) return 'bg-red-500';
        if (score >= 50) return 'bg-agent-yellow';
        if (score >= 30) return 'bg-agent-green';
        return 'bg-gray-400';
    };

    const scoreColor = getScoreColor(data.prospect_score);

    return (
        <div className="min-h-screen bg-gray-50 p-8">
            <div className="max-w-7xl mx-auto space-y-10">
                <header className="mb-8">
                    <button onClick={() => window.history.back()} className="text-agent-blue hover:text-agent-green transition duration-150 mb-4 font-semibold">
                        &larr; Back to History
                    </button>
                    <h1 className="text-4xl font-bold text-gray-800">{data.company_name} ({data.domain})</h1>
                    <p className="text-lg text-gray-500">Research ID: <span className="font-mono text-sm">{researchId}</span></p>
                </header>

                {/* --- 1. Score & Recommendation Card --- */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className={`col-span-1 p-6 rounded-xl shadow-2xl text-white ${scoreColor}`}>
                        <p className="text-sm uppercase font-light">Prospect Score (0-100)</p>
                        <p className="text-6xl font-extrabold mt-1">{data.prospect_score}</p>
                        <p className="text-xl font-semibold mt-4">{data.priority}</p>
                    </div>
                    <div className="col-span-2 bg-white p-6 rounded-xl shadow-lg border border-gray-200">
                        <h2 className="text-xl font-semibold text-agent-blue mb-3">Agent Recommendation</h2>
                        <p className="text-gray-700 text-lg italic">{data.recommendation}</p>
                        <div className="mt-4 pt-4 border-t border-gray-100">
                            <h3 className="text-md font-semibold text-gray-600">Signal Summary:</h3>
                            <ul className="flex flex-wrap gap-x-4 gap-y-2 mt-2">
                                {Object.entries(data.signal_summary || {}).filter(([k, v]) => v > 0).map(([key, count]) => (
                                    <li key={key} className="px-3 py-1 bg-gray-100 text-gray-700 text-sm rounded-full font-medium">
                                        {key.replace('_', ' ')} ({count})
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </div>

                {/* --- 2. Detected Signals --- */}
                <div className="pt-4">
                    <h2 className="text-3xl font-bold text-gray-800 mb-6">Detected Buying Signals ({data.total_signals})</h2>
                    {data.total_signals > 0 ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {data.signals.map((signal, index) => (
                                <SignalCard key={index} signal={signal} />
                            ))}
                        </div>
                    ) : (
                        <p className="text-gray-500">No strong buying signals detected in the latest news feed.</p>
                    )}
                </div>

                {/* --- 3. Full Research Data (Simple List) --- */}
                {/* Note: Contacts and Articles are stored in the original /company/{domain} route,
           but for simplicity, we show only the data available via the /signals/{id} route.
           In a production app, you'd fetch the full record here. */}
                <div className="pt-4">
                    <h2 className="text-3xl font-bold text-gray-800 mb-6">Raw Research Data (Contacts & Articles)</h2>
                    <p className="text-gray-500">For full article/contact list, please check the database or the main research response.</p>
                </div>

            </div>
        </div>
    );
};

export default DetailPage;