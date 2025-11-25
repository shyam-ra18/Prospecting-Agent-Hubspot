import React, { useState } from 'react';
import { API_BASE_URL } from '../const';

const ResearchForm = ({ onResearchStart }) => {
    const [domain, setDomain] = useState('');
    const [ticker, setTicker] = useState('');
    const [useVerified, setUseVerified] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    const handleResearch = async (e) => {
        e.preventDefault();
        if (!domain) {
            setError("Company domain is required.");
            return;
        }

        setIsLoading(true);
        setError('');

        const endpoint = useVerified ? 'research-verified' : 'research';
        const apiUrl = `${API_BASE_URL}/${endpoint}?domain=${domain}${ticker ? `&ticker=${ticker}` : ''}`;

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
            setError("Network error. Please check if the backend server is running.");
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="max-w-2xl mx-auto">
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-2">New Prospect Research</h2>
                <p className="text-gray-600">Analyze companies for expansion signals and outreach opportunities</p>
            </div>

            <form onSubmit={handleResearch} className="space-y-6">
                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Company Domain *
                            </label>
                            <input
                                type="text"
                                placeholder="example.com"
                                value={domain}
                                onChange={(e) => { setDomain(e.target.value); setError(''); }}
                                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                required
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Stock Ticker (Optional)
                            </label>
                            <input
                                type="text"
                                placeholder="TICKER"
                                value={ticker}
                                onChange={(e) => setTicker(e.target.value)}
                                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            />
                        </div>

                        <div className="flex items-center space-x-3">
                            <input
                                type="checkbox"
                                id="verified-research"
                                checked={useVerified}
                                onChange={(e) => setUseVerified(e.target.checked)}
                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                            />
                            <label htmlFor="verified-research" className="text-sm text-gray-700">
                                Use verified research (LLM validation + email verification)
                            </label>
                        </div>
                    </div>
                </div>

                <button
                    type="submit"
                    disabled={isLoading}
                    className={`w-full py-3 px-4 rounded-md text-white font-medium focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${isLoading
                        ? 'bg-gray-400 cursor-not-allowed'
                        : 'bg-blue-600 hover:bg-blue-700'
                        }`}
                >
                    {isLoading ? (
                        <div className="flex items-center justify-center space-x-2">
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                            <span>Analyzing Company Data...</span>
                        </div>
                    ) : (
                        `Start ${useVerified ? 'Verified ' : ''}Research`
                    )}
                </button>

                {error && (
                    <div className="p-4 bg-red-50 border border-red-200 rounded-md text-red-700 text-sm">
                        <strong>Error:</strong> {error}
                    </div>
                )}

                <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
                    <h4 className="font-medium text-blue-800 mb-2">Research Analysis Includes:</h4>
                    <ul className="text-blue-700 text-sm space-y-1">
                        <li>• Company firmographics and employee data</li>
                        <li>• Recent news and expansion signals</li>
                        <li>• Leadership changes and funding rounds</li>
                        <li>• Contact information for key decision makers</li>
                        {useVerified && (
                            <>
                                <li>• LLM-powered data validation</li>
                                <li>• Email verification and deliverability checks</li>
                            </>
                        )}
                    </ul>
                </div>
            </form>
        </div>
    );
};

export default ResearchForm;