import React, { useState } from 'react';
import ResearchForm from '../components/ResearchForm';
import ProspectHistory from '../components/ProspectHistory';
import SystemMetrics from '../components/SystemMetrics';

const HomePage = () => {
    const [newResearchData, setNewResearchData] = useState(null);
    const [activeTab, setActiveTab] = useState('research');

    const handleResearchStart = (data) => {
        setNewResearchData(data);
        setActiveTab('history');
    };

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <header className="bg-white border-b border-gray-200">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center py-6">
                        <div className="flex items-center space-x-3">
                            <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                                <span className="text-white font-bold text-lg">DL</span>
                            </div>
                            <div>
                                <h1 className="text-2xl font-bold text-gray-900">Dublabs.ai Prospect</h1>
                                <p className="text-sm text-gray-500">AI-powered sales intelligence platform</p>
                            </div>
                        </div>
                        <div className="text-sm text-gray-500">
                            v4.0 • Verified Edition
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {/* Navigation Tabs */}
                <div className="mb-8">
                    <div className="border-b border-gray-200">
                        <nav className="-mb-px flex space-x-8">
                            {[
                                { id: 'research', label: 'New Research', icon: '🔍' },
                                { id: 'history', label: 'Prospect History', icon: '📊' },
                                { id: 'metrics', label: 'System Analytics', icon: '📈' }
                            ].map(tab => (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm ${activeTab === tab.id
                                            ? 'border-blue-500 text-blue-600'
                                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                        }`}
                                >
                                    <span className="mr-2">{tab.icon}</span>
                                    {tab.label}
                                </button>
                            ))}
                        </nav>
                    </div>
                </div>

                {/* Content Sections */}
                <div className="bg-white rounded-lg shadow-sm border border-gray-200">
                    {activeTab === 'research' && (
                        <div className="p-6">
                            <ResearchForm onResearchStart={handleResearchStart} />
                        </div>
                    )}

                    {activeTab === 'history' && (
                        <div className="p-6">
                            <ProspectHistory newResearchData={newResearchData} />
                        </div>
                    )}

                    {activeTab === 'metrics' && (
                        <div className="p-6">
                            <SystemMetrics />
                        </div>
                    )}
                </div>
            </main>

            {/* Footer */}
            <footer className="bg-white border-t border-gray-200 mt-12">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                    <div className="text-center text-gray-500 text-sm">
                        <div className="flex items-center justify-center space-x-6 mb-3">
                            <span className="flex items-center">
                                <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                                System Operational
                            </span>
                            <span>•</span>
                            <span>FastAPI + React</span>
                            <span>•</span>
                            <span>Real-time Analytics</span>
                        </div>
                        <p>Dublabs.ai Prospecting Platform • Advanced lead scoring and outreach automation</p>
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default HomePage;