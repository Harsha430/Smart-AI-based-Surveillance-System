import React from 'react';

export default function VideoFeed(){
  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-b-lg border-t border-white/5 bg-black">
      <img
        src="/video_feed"
        alt="Live Feed"
        className="w-full h-full object-contain select-none"
        draggable={false}
        style={{background:'radial-gradient(circle at center, #111 0%, #000 60%)'}}
      />
      <div className="absolute top-2 right-2 text-[10px] font-mono bg-red-600/80 text-white px-2 py-0.5 rounded animate-pulse">LIVE</div>
    </div>
  );
}

