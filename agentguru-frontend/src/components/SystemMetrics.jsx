import React, { useState, useEffect } from 'react';

const SystemMetrics = () => {
    const [systemMetrics, setSystemMetrics] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchSystemMetrics = async () => {
            try {
                const response = await fetch('http://127.0.0.1:8000/metrics/evaluate/system');
                const data = await response.json();

                if (response.ok) {
                    setSystemMetrics(data);
                } else {
                    setError(data.detail || 'Failed to fetch system metrics');
                }
            } catch (err) {
                setError('Network error. Could not connect to backend.');
            } finally {
                setIsLoading(false);
            }
        };

        fetchSystemMetrics();
    }, []);

    if (isLoading) return (
        <div className="bg-white p-8 rounded-2xl shadow-lg border border-gray-200 text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-agent-blue mx-auto mb-4"></div>
            <p className="text-gray-600">Loading system metrics...</p>
        </div>
    );

    if (error) return (
        <div className="bg-white p-8 rounded-2xl shadow-lg border border-red-200 text-center">
            <div className="text-red-500 text-4xl mb-4">⚠️</div>
            <h3 className="text-xl font-semibold text-red-600 mb-2">Error</h3>
            <p className="text-gray-600">{error}</p>
        </div>
    );

    return (
        <div className="space-y-6">
            <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-200">
                <h2 className="text-2xl font-bold text-agent-blue mb-2">📈 System Performance Overview</h2>
                <p className="text-gray-600">Real-time metrics and performance analytics</p>
            </div>

            {systemMetrics?.performance && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <div className="bg-gradient-to-br from-blue-500 to-blue-600 p-6 rounded-2xl text-white shadow-lg">
                        <p className="text-sm opacity-90">Total Researches</p>
                        <p className="text-3xl font-bold mt-2">{systemMetrics.performance.total_researches_analyzed}</p>
                    </div>
                    <div className="bg-gradient-to-br from-green-500 to-green-600 p-6 rounded-2xl text-white shadow-lg">
                        <p className="text-sm opacity-90">Success Rate</p>
                        <p className="text-3xl font-bold mt-2">{systemMetrics.performance.success_rate_percent}%</p>
                    </div>
                    <div className="bg-gradient-to-br from-purple-500 to-purple-600 p-6 rounded-2xl text-white shadow-lg">
                        <p className="text-sm opacity-90">Avg Processing Time</p>
                        <p className="text-3xl font-bold mt-2">{systemMetrics.performance.average_processing_time_ms}ms</p>
                    </div>
                    <div className="bg-gradient-to-br from-orange-500 to-orange-600 p-6 rounded-2xl text-white shadow-lg">
                        <p className="text-sm opacity-90">Avg Prospect Score</p>
                        <p className="text-3xl font-bold mt-2">{systemMetrics.performance.average_prospect_score}</p>
                    </div>
                </div>
            )}

            {systemMetrics?.data_quality && (
                <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-200">
                    <h3 className="text-xl font-semibold text-gray-800 mb-4">📊 Data Quality Metrics</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="text-center p-4 bg-blue-50 rounded-xl border border-blue-200">
                            <p className="text-2xl font-bold text-blue-600">{systemMetrics.data_quality.average_contacts_per_company}</p>
                            <p className="text-sm text-gray-600 mt-1">Avg Contacts/Company</p>
                        </div>
                        <div className="text-center p-4 bg-green-50 rounded-xl border border-green-200">
                            <p className="text-2xl font-bold text-green-600">{systemMetrics.data_quality.average_articles_per_company}</p>
                            <p className="text-sm text-gray-600 mt-1">Avg Articles/Company</p>
                        </div>
                        <div className="text-center p-4 bg-purple-50 rounded-xl border border-purple-200">
                            <p className="text-2xl font-bold text-purple-600">{systemMetrics.data_quality.companies_with_contacts}%</p>
                            <p className="text-sm text-gray-600 mt-1">Companies with Contacts</p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default SystemMetrics;