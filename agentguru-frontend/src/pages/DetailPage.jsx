import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import SignalCard from '../components/SignalCard';
import OutreachComposerCard from '../components/OutreachComposerCard';
import MetricsDashboard from '../components/MetricsDashboard';

const DetailPage = () => {
    const { researchId } = useParams();
    const [data, setData] = useState(null);
    const [fullRecord, setFullRecord] = useState(null);
    const [metrics, setMetrics] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');
    const [activeSection, setActiveSection] = useState('overview');

    useEffect(() => {
        const fetchAllData = async () => {
            setIsLoading(true);
            setError('');
            try {
                // Fetch Signals & Score
                const signalUrl = `http://127.0.0.1:8000/signals/${researchId}`;
                const signalResponse = await fetch(signalUrl);
                const signalJson = await signalResponse.json();

                if (!signalResponse.ok) {
                    throw new Error(signalJson.detail || "Failed to fetch signal data.");
                }
                setData(signalJson);
                setMetrics(signalJson.metrics);

                // Fetch Full Company Record
                const domain = signalJson.domain;
                if (domain) {
                    const fullRecordUrl = `http://127.0.0.1:8000/company/${domain}`;
                    const fullRecordResponse = await fetch(fullRecordUrl);
                    const fullRecordJson = await fullRecordResponse.json();

                    if (fullRecordResponse.ok) {
                        setFullRecord(fullRecordJson);
                    }
                }

            } catch (err) {
                setError(err.message || "Network error. Could not connect to the backend.");
            } finally {
                setIsLoading(false);
            }
        };
        fetchAllData();
    }, [researchId]);

    if (isLoading) return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center">
            <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <p className="text-lg font-medium text-gray-700">Analyzing prospect data...</p>
                <p className="text-gray-500 mt-2">Processing company signals and metrics</p>
            </div>
        </div>
    );

    if (error) return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center">
            <div className="bg-white p-8 rounded-lg shadow-sm border border-gray-200 max-w-md text-center">
                <div className="text-red-500 text-4xl mb-4">⚠️</div>
                <h2 className="text-xl font-semibold text-gray-900 mb-2">Error Loading Data</h2>
                <p className="text-gray-600 mb-6">{error}</p>
                <button
                    onClick={() => window.history.back()}
                    className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-sm font-medium"
                >
                    Return to Dashboard
                </button>
            </div>
        </div>
    );

    if (!data) return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center">
            <div className="text-center">
                <p className="text-gray-700">No data found for this research.</p>
            </div>
        </div>
    );

    const getScoreColor = (score) => {
        if (score >= 75) return 'bg-gradient-to-r from-red-500 to-orange-500';
        if (score >= 60) return 'bg-gradient-to-r from-yellow-500 to-amber-500';
        if (score >= 40) return 'bg-gradient-to-r from-green-500 to-emerald-500';
        return 'bg-gradient-to-r from-gray-400 to-gray-500';
    };

    const scoreColor = getScoreColor(data.prospect_score);
    const contacts = fullRecord?.contacts || [];
    const articles = fullRecord?.articles || [];

    const navigationSections = [
        { id: 'overview', label: 'Overview', icon: '📊' },
        { id: 'signals', label: 'Buying Signals', icon: '🎯' },
        { id: 'outreach', label: 'Outreach', icon: '✉️' },
        { id: 'metrics', label: 'Analytics', icon: '📈' },
        { id: 'research', label: 'Research Data', icon: '🔍' }
    ];

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center justify-between py-4">
                        <div className="flex items-center space-x-4">
                            <button
                                onClick={() => window.history.back()}
                                className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 text-sm font-medium"
                            >
                                <span>←</span>
                                <span>Back</span>
                            </button>
                            <div className="h-4 w-px bg-gray-300"></div>
                            <div>
                                <h1 className="text-xl font-semibold text-gray-900">{data.company_name}</h1>
                                <p className="text-sm text-gray-500">{data.domain}</p>
                            </div>
                        </div>
                        <div className="flex items-center space-x-4">
                            {data.is_verified && (
                                <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full font-medium border border-green-200">
                                    ✓ Verified
                                </span>
                            )}
                            <div className={`px-4 py-2 rounded-lg ${scoreColor} text-white font-semibold text-lg`}>
                                {data.prospect_score}
                            </div>
                        </div>
                    </div>

                    {/* Navigation */}
                    <div className="border-b border-gray-200">
                        <nav className="-mb-px flex space-x-8 overflow-x-auto">
                            {navigationSections.map(section => (
                                <button
                                    key={section.id}
                                    onClick={() => setActiveSection(section.id)}
                                    className={`whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm ${activeSection === section.id
                                            ? 'border-blue-500 text-blue-600'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                        }`}
                                >
                                    <span className="mr-2">{section.icon}</span>
                                    {section.label}
                                </button>
                            ))}
                        </nav>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
                {/* Overview Section */}
                {activeSection === 'overview' && (
                    <div className="space-y-6">
                        {/* Score & Priority */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            <div className={`p-6 rounded-lg shadow-sm border border-gray-200 text-white ${scoreColor}`}>
                                <p className="text-sm font-medium opacity-90">Prospect Score</p>
                                <p className="text-4xl font-bold mt-2">{data.prospect_score}<span className="text-xl font-light opacity-90">/100</span></p>
                                <div className="flex items-center space-x-2 mt-4">
                                    <span className="text-lg">{data.priority.split(' ')[0]}</span>
                                    <span className="font-medium">{data.priority}</span>
                                </div>
                            </div>

                            <div className="lg:col-span-2 bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                                <h2 className="text-lg font-semibold text-gray-900 mb-4">Agent Recommendation</h2>
                                <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
                                    <p className="text-gray-800 leading-relaxed whitespace-pre-line">{data.recommendation}</p>
                                </div>

                                <div className="mt-6 pt-6 border-t border-gray-100">
                                    <h3 className="text-sm font-medium text-gray-700 mb-3">Signal Summary</h3>
                                    <div className="flex flex-wrap gap-2">
                                        {Object.entries(data?.signal_summary || {})
                                            .filter(([k, v]) => v > 0)
                                            .map(([key, count]) => (
                                                <span key={key} className="px-3 py-1 bg-gray-100 text-gray-700 text-sm rounded-md font-medium border border-gray-300">
                                                    {key.replace(/_/g, ' ')} ({count})
                                                </span>
                                            ))}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Quick Metrics */}
                        {metrics && (
                            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                                <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Metrics</h2>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <div className="text-center p-4 bg-gray-50 rounded-md border border-gray-200">
                                        <p className="text-2xl font-bold text-gray-900">{metrics.overall_score}</p>
                                        <p className="text-sm text-gray-600 mt-1">Overall Score</p>
                                    </div>
                                    <div className="text-center p-4 bg-gray-50 rounded-md border border-gray-200">
                                        <p className="text-2xl font-bold text-gray-900">{metrics.functionality?.functionality_score}</p>
                                        <p className="text-sm text-gray-600 mt-1">Functionality</p>
                                    </div>
                                    <div className="text-center p-4 bg-gray-50 rounded-md border border-gray-200">
                                        <p className="text-2xl font-bold text-gray-900">{metrics.security?.security_score}</p>
                                        <p className="text-sm text-gray-600 mt-1">Security</p>
                                    </div>
                                    <div className="text-center p-4 bg-gray-50 rounded-md border border-gray-200">
                                        <p className="text-2xl font-bold text-gray-900">{metrics.latency?.latency_score}</p>
                                        <p className="text-sm text-gray-600 mt-1">Performance</p>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Breakdown Scores */}
                        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                            <h2 className="text-lg font-semibold text-gray-900 mb-4">Score Breakdown</h2>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                {Object.entries(data.breakdown || {}).map(([key, value]) => (
                                    <div key={key} className="text-center p-4 bg-gray-50 rounded-md border border-gray-200">
                                        <p className="text-sm font-medium text-gray-600 mb-2">{key.replace(/_/g, ' ')}</p>
                                        <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                                            <div
                                                className="bg-blue-600 h-2 rounded-full transition-all duration-1000"
                                                style={{ width: `${value}%` }}
                                            ></div>
                                        </div>
                                        <p className="text-xl font-semibold text-gray-900">{value}%</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                {/* Signals Section */}
                {activeSection === 'signals' && (
                    <div>
                        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 mb-6">
                            <h2 className="text-xl font-semibold text-gray-900 mb-2">Detected Buying Signals</h2>
                            <p className="text-gray-600">Found {data.total_signals} significant signals in recent activities</p>
                        </div>

                        {data.total_signals > 0 ? (
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                                {data.signals?.map((signal, index) => (
                                    <SignalCard key={index} signal={signal} />
                                ))}
                            </div>
                        ) : (
                            <div className="bg-white p-8 rounded-lg shadow-sm border border-gray-200 text-center">
                                <div className="text-4xl mb-4">🔍</div>
                                <h3 className="text-lg font-semibold text-gray-700 mb-2">No Strong Signals Detected</h3>
                                <p className="text-gray-500">No significant buying signals were found in the recent analysis.</p>
                            </div>
                        )}
                    </div>
                )}

                {/* Outreach Section */}
                {activeSection === 'outreach' && (
                    <div>
                        <OutreachComposerCard researchId={researchId} />
                    </div>
                )}

                {/* Metrics Section */}
                {activeSection === 'metrics' && (
                    <div>
                        <MetricsDashboard metrics={metrics} researchData={data} />
                    </div>
                )}

                {/* Research Data Section */}
                {activeSection === 'research' && (
                    <div className="space-y-6">
                        {/* Contacts & Articles */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                                <div className="flex items-center justify-between mb-4">
                                    <h2 className="text-lg font-semibold text-gray-900">Target Contacts</h2>
                                    <span className="px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded-md font-medium border border-blue-200">
                                        {contacts.length} contacts
                                    </span>
                                </div>
                                <div className="space-y-3 max-h-96 overflow-y-auto">
                                    {contacts.length > 0 ? (
                                        contacts.map((contact, index) => (
                                            <div key={index} className="p-4 border border-gray-200 rounded-md hover:border-blue-300 transition-colors">
                                                <div className="flex items-start justify-between">
                                                    <div>
                                                        <p className="font-medium text-gray-900">{contact.name || 'Unknown Contact'}</p>
                                                        <p className="text-blue-600 text-sm font-medium mt-1">{contact.title || 'Title N/A'}</p>
                                                        <p className="text-gray-500 text-sm mt-1">{contact.email || 'Email N/A'}</p>
                                                    </div>
                                                    {contact.confidence && (
                                                        <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-md font-medium">
                                                            {contact.confidence}% confidence
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        ))
                                    ) : (
                                        <div className="text-center py-8 text-gray-500">
                                            <p>No contacts found for this domain</p>
                                        </div>
                                    )}
                                </div>
                            </div>

                            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                                <div className="flex items-center justify-between mb-4">
                                    <h2 className="text-lg font-semibold text-gray-900">News Articles Analyzed</h2>
                                    <span className="px-3 py-1 bg-green-100 text-green-800 text-sm rounded-md font-medium border border-green-200">
                                        {articles.length} articles
                                    </span>
                                </div>
                                <div className="space-y-3 max-h-96 overflow-y-auto">
                                    {articles.length > 0 ? (
                                        articles.map((article, index) => (
                                            <a
                                                key={index}
                                                href={article.url || article.link}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="block p-4 border border-gray-200 rounded-md hover:border-green-300 hover:shadow-sm transition-all"
                                            >
                                                <p className="font-medium text-gray-900 line-clamp-2 mb-2">{article.title}</p>
                                                <div className="flex items-center justify-between text-sm text-gray-500">
                                                    <span>{article.source}</span>
                                                    <span>{article.date || article.time_published}</span>
                                                </div>
                                            </a>
                                        ))
                                    ) : (
                                        <div className="text-center py-8 text-gray-500">
                                            <p>No news articles found for this research</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Fit Analysis */}
                        {data.fit_analysis && (
                            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                                <h2 className="text-lg font-semibold text-gray-900 mb-4">Fit Analysis</h2>
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                                    <div className="text-center p-4 bg-gray-50 rounded-md border border-gray-200">
                                        <p className="text-sm text-gray-600 mb-1">Industry Match</p>
                                        <p className={`text-lg font-semibold ${data.fit_analysis.industry_match ? 'text-green-600' : 'text-red-600'}`}>
                                            {data.fit_analysis.industry_match ? '✓ Match' : '✗ No Match'}
                                        </p>
                                    </div>
                                    <div className="text-center p-4 bg-gray-50 rounded-md border border-gray-200">
                                        <p className="text-sm text-gray-600 mb-1">Size Match</p>
                                        <p className={`text-lg font-semibold ${data.fit_analysis.size_match ? 'text-green-600' : 'text-orange-600'}`}>
                                            {data.fit_analysis.size_match ? '✓ Ideal' : 'Acceptable'}
                                        </p>
                                    </div>
                                    <div className="text-center p-4 bg-gray-50 rounded-md border border-gray-200">
                                        <p className="text-sm text-gray-600 mb-1">Employees</p>
                                        <p className="text-lg font-semibold text-gray-900">{data.fit_analysis.employee_count || 'N/A'}</p>
                                    </div>
                                    <div className="text-center p-4 bg-gray-50 rounded-md border border-gray-200">
                                        <p className="text-sm text-gray-600 mb-1">Contacts</p>
                                        <p className="text-lg font-semibold text-gray-900">{data.fit_analysis.contact_count}</p>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default DetailPage;