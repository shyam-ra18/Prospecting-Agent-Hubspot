import React from 'react';

const SignalCard = ({ signal }) => {
    // Simple color logic based on signal strength
    const getColor = (weight) => {
        if (weight >= 22) return 'bg-red-100 text-red-800 border-red-300';
        if (weight >= 18) return 'bg-agent-yellow/20 text-agent-yellow border-agent-yellow/50';
        return 'bg-green-100 text-agent-green border-agent-green/50';
    };

    const colorClass = getColor(signal.weight);

    return (
        <a href={signal.url} target="_blank" rel="noopener noreferrer"
            className={`block p-4 rounded-lg shadow-sm border ${colorClass} transition transform hover:shadow-md hover:scale-[1.01] cursor-pointer`}>
            <p className="text-xs font-semibold uppercase">{signal.type.replace('_', ' ')} (Wt: {signal.weight})</p>
            <h3 className="text-sm font-bold mt-1 leading-snug line-clamp-2">{signal.title}</h3>
            <p className="text-xs mt-2 opacity-80">Source: {signal.source} | Date: {signal.date}</p>
        </a>
    );
};

export default SignalCard;