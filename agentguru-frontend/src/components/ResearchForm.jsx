import React, { useState } from 'react';

const ResearchForm = ({ onResearchStart }) => {
    const [domain, setDomain] = useState('');
    const [ticker, setTicker] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    const handleResearch = async (e) => {
        e.preventDefault();
        if (!domain) {
            setError("Domain is required.");
            return;
        }

        setIsLoading(true);
        setError('');

        const apiUrl = `http://127.0.0.1:8000/research?domain=${domain}${ticker ? `&ticker=${ticker}` : ''}`;

        try {
            const response = await fetch(apiUrl, { method: 'POST' });
            const data = await response.json();

            if (response.ok) {
                onResearchStart(data);
                setDomain('');
                setTicker('');
            } else {
                setError(data.detail || "Failed to start research.");
            }
        } catch (err) {
            setError("Network error. Is the FastAPI backend running?");
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-100">
            <h2 className="text-xl font-semibold text-agent-blue mb-4">Start New Prospect Research</h2>
            <form onSubmit={handleResearch} className="space-y-4">
                <input
                    type="text"
                    placeholder="Company Domain (e.g., apple.com)"
                    value={domain}
                    onChange={(e) => { setDomain(e.target.value); setError(''); }}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-agent-blue focus:border-agent-blue"
                    required
                />
                <input
                    type="text"
                    placeholder="Stock Ticker (e.g., AAPL) - Optional"
                    value={ticker}
                    onChange={(e) => setTicker(e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-agent-blue focus:border-agent-blue"
                />
                <button
                    type="submit"
                    className={`w-full py-3 rounded-lg text-white font-bold transition duration-200 ${isLoading ? 'bg-gray-400 cursor-not-allowed' : 'bg-agent-green hover:bg-emerald-600'
                        }`}
                    disabled={isLoading}
                >
                    {isLoading ? 'Researching...' : 'Run Prospecting Agent 🚀'}
                </button>
                {error && <p className="text-red-500 text-sm">{error}</p>}
            </form>
        </div>
    );
};

export default ResearchForm;