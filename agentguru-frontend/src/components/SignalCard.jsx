import React from 'react';

const SignalCard = ({ signal }) => {
    const getSignalStyles = (weight, category) => {
        const baseStyles = "block p-4 rounded-md border transition-colors cursor-pointer ";

        if (category === 'negative') {
            return baseStyles + "bg-red-50 border-red-200 text-red-800 hover:border-red-300";
        }

        if (weight >= 35) {
            return baseStyles + "bg-red-50 border-red-200 text-red-800 hover:border-red-300";
        } else if (weight >= 20) {
            return baseStyles + "bg-orange-50 border-orange-200 text-orange-800 hover:border-orange-300";
        } else if (weight >= 10) {
            return baseStyles + "bg-yellow-50 border-yellow-200 text-yellow-800 hover:border-yellow-300";
        } else {
            return baseStyles + "bg-green-50 border-green-200 text-green-800 hover:border-green-300";
        }
    };

    const getWeightBadge = (weight) => {
        if (weight >= 35) return { label: 'CRITICAL', color: 'bg-red-500 text-white' };
        if (weight >= 20) return { label: 'HIGH', color: 'bg-orange-500 text-white' };
        if (weight >= 10) return { label: 'MEDIUM', color: 'bg-yellow-500 text-white' };
        return { label: 'LOW', color: 'bg-green-500 text-white' };
    };

    const weightBadge = getWeightBadge(signal.weight);

    return (
        <a
            href={signal.url}
            target="_blank"
            rel="noopener noreferrer"
            className={getSignalStyles(signal.weight, signal.category)}
        >
            <div className="flex items-start justify-between mb-2">
                <span className="text-xs font-medium uppercase tracking-wide opacity-80">
                    {signal.type.replace(/_/g, ' ')}
                </span>
                <span className={`px-2 py-1 text-xs font-semibold rounded ${weightBadge.color}`}>
                    {weightBadge.label}
                </span>
            </div>

            <h3 className="font-medium text-sm leading-snug line-clamp-3 mb-2">
                {signal.title}
            </h3>

            <div className="flex items-center justify-between text-xs text-gray-600">
                <span>{signal.source}</span>
                <span>{signal.date}</span>
            </div>

            {signal.days_old && (
                <div className="mt-2 flex items-center justify-between text-xs">
                    <span className="text-gray-500">{signal.days_old} days ago</span>
                    <span className="font-medium">
                        {signal.decayed_weight || signal.weight} pts
                    </span>
                </div>
            )}
        </a>
    );
};

export default SignalCard;