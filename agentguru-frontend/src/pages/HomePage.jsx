import React, { useState } from 'react';
import ResearchForm from '../components/ResearchForm';
import ProspectHistory from '../components/ProspectHistory';

const HomePage = () => {
    const [newResearchData, setNewResearchData] = useState(null);

    const handleResearchStart = (data) => {
        setNewResearchData(data);
    };

    return (
        <div className="min-h-screen bg-gray-50 p-8">
            <header className="text-center mb-10">
                <h1 className="text-4xl font-extrabold text-agent-blue">AgentGuru Prospecting Dashboard 📊</h1>
                <p className="mt-2 text-xl text-gray-600">Quick & Dirty Lead Scoring Prototype</p>
            </header>

            <div className="max-w-6xl mx-auto space-y-8">
                <ResearchForm onResearchStart={handleResearchStart} />

                <hr className="border-gray-300" />

                <ProspectHistory newResearchData={newResearchData} />
            </div>

            <footer className="text-center mt-12 text-gray-500 text-sm">
                Data is user-generated and unverified. Prototype powered by FastAPI, MongoDB, React & Tailwind.
            </footer>
        </div>
    );
};

export default HomePage;