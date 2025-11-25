import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const ProspectHistory = ({ newResearchData }) => {
    const navigate = useNavigate();
    const [history, setHistory] = useState([]);
    const [isLoading, setIsLoading] = useState(true);

    // Helper to get the color for the score pill
    const getScoreColor = (score) => {
        if (score >= 80) return 'bg-red-500';
        if (score >= 55) return 'bg-agent-yellow';
        if (score >= 30) return 'bg-agent-green';
        return 'bg-gray-400';
    };


    /**
     * Fetches the list of all companies from the new /companies endpoint.
     * This replaces the inefficient mock domain iteration.
     */
    const fetchHistory = async () => {
        setIsLoading(true);
        try {
            // 1. Fetch the list of companies from the new dedicated endpoint
            const response = await fetch(`http://127.0.0.1:8000/companies?limit=20`); // Request the top 20 recent

            if (response.ok) {
                let fetchedData = await response.json();

                // 2. Handle the case where the new research needs to be prepended
                if (newResearchData) {
                    const newId = newResearchData.research_id;

                    // If the new research is not in the list (because it's the very latest)
                    if (!fetchedData.find(d => d.research_id === newId)) {

                        // We need the full details, which are available via /company/{domain}
                        const fullRecordResponse = await fetch(`http://127.0.0.1:8000/company/${newResearchData.domain}`);

                        if (fullRecordResponse.ok) {
                            let newRecord = await fullRecordResponse.json();
                            // Manually add the research_id back since /company/{domain} pops _id
                            newRecord.research_id = newId;
                            fetchedData.unshift(newRecord);
                        }
                    }
                }

                // Ensure sorting (though the API should handle it)
                fetchedData.sort((a, b) => new Date(b.researched_at) - new Date(a.researched_at));
                setHistory(fetchedData);
            } else {
                console.error("API Error:", await response.json());
                setHistory([]);
            }
        } catch (e) {
            console.error("Network or Fetch Error:", e);
            setHistory([]);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchHistory();
    }, [newResearchData]); // Re-fetch when new research is initiated

    if (isLoading) return <p className="text-center text-gray-500">Loading research history...</p>;

    return (
        <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-100">
            <h2 className="text-xl font-semibold text-agent-blue mb-4">Recent Prospect Research History (Top 20)</h2>
            {history.length === 0 ? (
                <p className="text-gray-500">No research history found. Run a search above!</p>
            ) : (
                <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                        <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Company</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Industry</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Score</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                            <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {history.map((item) => (
                            // Use research_id for the key as it is now guaranteed to be a unique string
                            <tr key={item.research_id || item.domain} className="hover:bg-gray-50">
                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{item.company_name} ({item.domain})</td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.industry || 'N/A'}</td>
                                <td className="px-6 py-4 whitespace-nowrap">
                                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full text-white ${getScoreColor(item.prospect_score)}`}>
                                        {item.prospect_score || 'N/A'}
                                    </span>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{new Date(item.researched_at).toLocaleDateString()}</td>
                                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                    <button
                                        // Navigate using the research_id (UUID)
                                        onClick={() => navigate(`/detail/${item.research_id}`)}
                                        className="text-agent-green hover:text-emerald-600 font-semibold"
                                    >
                                        View Detail →
                                    </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </div>
    );
};

export default ProspectHistory;