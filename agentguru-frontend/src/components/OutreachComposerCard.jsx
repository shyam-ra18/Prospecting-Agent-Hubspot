import React, { useState, useEffect } from 'react';
import { API_BASE_URL } from '../const';

const OutreachComposerCard = ({ researchId }) => {
    const [outreachData, setOutreachData] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');
    const [activeTab, setActiveTab] = useState('email_1_initial_outreach');

    useEffect(() => {
        const fetchOutreach = async () => {
            setIsLoading(true);
            setError('');
            try {
                const url = `${API_BASE_URL}/compose/${researchId}`;
                const response = await fetch(url);
                const json = await response.json();

                if (response.ok) {
                    setOutreachData(json);
                    // Ensure the first email is set as active tab
                    const firstEmail = Object.keys(json.sequence)[0];
                    setActiveTab(firstEmail);
                } else {
                    setError(json.detail || "Failed to compose outreach sequence.");
                }
            } catch (err) {
                setError("Network error. Could not connect to outreach endpoint.");
            } finally {
                setIsLoading(false);
            }
        };
        fetchOutreach();
    }, [researchId]);

    if (isLoading) return <div className="p-4 text-center text-sm text-gray-500">Composing personalized outreach... 📧</div>;
    if (error) return <div className="p-4 text-center text-red-600 text-sm">Error: {error}</div>;
    if (!outreachData) return null;

    const sequence = outreachData.sequence;
    const currentEmail = sequence[activeTab];
    const contact = outreachData.target_contact;

    // Helper to format the tab name
    const formatTabName = (key) => key.split('_').slice(2).map(s => s.charAt(0).toUpperCase() + s.slice(1)).join(' ');


    return (
        <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
            <h2 className="text-2xl font-bold text-agent-blue mb-4">Personalized Outreach Sequence (Dublab.sh)</h2>

            <div className="flex border-b border-gray-200 mb-4 overflow-x-auto">
                {Object.keys(sequence).map((key) => (
                    <button
                        key={key}
                        onClick={() => setActiveTab(key)}
                        className={`py-2 px-4 text-sm font-medium transition-colors duration-150 ${activeTab === key
                            ? 'border-b-2 border-agent-green text-agent-green'
                            : 'text-gray-500 hover:text-gray-700'
                            }`}
                    >
                        {formatTabName(key)}
                    </button>
                ))}
            </div>

            {currentEmail && (
                <div className="space-y-3">
                    <p className="font-mono text-xs text-gray-600">To: {contact.email || '[Email not found]'} ({contact.name || 'Target'})</p>
                    <p className="font-semibold text-lg">Subject: <span className="text-gray-800">{currentEmail.subject}</span></p>
                    <div className="border border-gray-100 p-4 bg-gray-50 rounded-lg whitespace-pre-wrap">
                        {/* Displaying the body with line breaks */}
                        <p className="text-gray-700 text-sm leading-relaxed">{currentEmail.body}</p>
                    </div>
                    <p className="mt-4 text-sm text-gray-600 font-medium">
                        Target Contact: <span className="text-agent-blue">{contact.name}</span>, {contact.title || 'N/A'}
                    </p>
                </div>
            )}
        </div>
    );
};

export default OutreachComposerCard;