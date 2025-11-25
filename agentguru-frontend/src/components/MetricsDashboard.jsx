import React from 'react';

const MetricsDashboard = ({ metrics, researchData }) => {
    if (!metrics) {
        return (
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 text-center">
                <h3 className="text-lg font-medium text-gray-700 mb-2">No Metrics Available</h3>
                <p className="text-gray-500">Metrics data is not available for this research.</p>
            </div>
        );
    }

    const MetricCard = ({ title, score, breakdown, color = 'blue' }) => {
        const colorClasses = {
            blue: 'from-blue-500 to-blue-600',
            green: 'from-green-500 to-green-600',
            purple: 'from-purple-500 to-purple-600',
            orange: 'from-orange-500 to-orange-600'
        };

        return (
            <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
                <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-medium text-gray-900">{title}</h3>
                    <div className={`px-2 py-1 bg-gradient-to-r ${colorClasses[color]} text-white rounded text-xs font-semibold`}>
                        {score}/10
                    </div>
                </div>
                <div className="space-y-2">
                    {breakdown && Object.entries(breakdown).map(([key, value]) => (
                        <div key={key} className="flex items-center justify-between">
                            <span className="text-xs text-gray-600 capitalize">{key.replace(/_/g, ' ')}</span>
                            <div className="flex items-center space-x-2">
                                <div className="w-16 bg-gray-200 rounded-full h-1.5">
                                    <div
                                        className={`bg-gradient-to-r ${colorClasses[color]} h-1.5 rounded-full`}
                                        style={{ width: `${value * 10}%` }}
                                    ></div>
                                </div>
                                <span className="text-xs font-medium text-gray-700 w-6">{value}</span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    return (
        <div className="space-y-6">
            {/* Overall Score */}
            <div className="bg-gradient-to-r from-blue-600 to-blue-700 p-6 rounded-lg text-white">
                <div className="flex items-center justify-between">
                    <div>
                        <h2 className="text-xl font-semibold mb-1">Overall System Score</h2>
                        <p className="opacity-90 text-sm">Comprehensive evaluation of research quality</p>
                    </div>
                    <div className="text-right">
                        <p className="text-3xl font-bold">{metrics.overall_score}<span className="text-lg font-light opacity-90">/10</span></p>
                        <p className="text-sm font-medium mt-1">
                            {metrics.overall_score >= 8 ? 'Excellent' :
                                metrics.overall_score >= 6 ? 'Good' :
                                    metrics.overall_score >= 4 ? 'Average' : 'Needs Improvement'}
                        </p>
                    </div>
                </div>
            </div>

            {/* Main Metrics Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <MetricCard
                    title="Functionality"
                    score={metrics.functionality?.functionality_score}
                    breakdown={metrics.functionality?.breakdown}
                    color="blue"
                />
                <MetricCard
                    title="Security"
                    score={metrics.security?.security_score}
                    breakdown={metrics.security?.breakdown}
                    color="green"
                />
                <MetricCard
                    title="Performance"
                    score={metrics.latency?.latency_score}
                    breakdown={metrics.latency?.breakdown}
                    color="purple"
                />
                <MetricCard
                    title="Research Quality"
                    score={Math.round((metrics.research_quality?.research_quality || 0) / 10)}
                    breakdown={metrics.research_quality}
                    color="orange"
                />
            </div>

            {/* Recommendations */}
            {metrics.recommendations && metrics.recommendations.length > 0 && (
                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                    <h3 className="text-lg font-medium text-gray-900 mb-3">Improvement Recommendations</h3>
                    <div className="space-y-2">
                        {metrics.recommendations.map((rec, index) => (
                            <div key={index} className="flex items-start space-x-3 p-3 bg-yellow-50 rounded-md border border-yellow-200">
                                <span className="text-yellow-600 mt-0.5">•</span>
                                <p className="text-sm text-gray-700">{rec}</p>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Processing Info */}
            {researchData.processing_time_ms && (
                <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                    <h3 className="text-lg font-medium text-gray-900 mb-3">Processing Information</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="text-center p-4 bg-gray-50 rounded-md border border-gray-200">
                            <p className="text-xl font-semibold text-gray-900">{researchData.processing_time_ms}ms</p>
                            <p className="text-sm text-gray-600 mt-1">Processing Time</p>
                        </div>
                        <div className="text-center p-4 bg-gray-50 rounded-md border border-gray-200">
                            <p className="text-xl font-semibold text-gray-900">{researchData.articles_analyzed || 0}</p>
                            <p className="text-sm text-gray-600 mt-1">Articles Analyzed</p>
                        </div>
                        <div className="text-center p-4 bg-gray-50 rounded-md border border-gray-200">
                            <p className="text-xl font-semibold text-gray-900">{researchData.total_signals || 0}</p>
                            <p className="text-sm text-gray-600 mt-1">Signals Detected</p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default MetricsDashboard;