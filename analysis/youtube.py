"""
YouTube Analytics Engine — Video View & Creator Market Predictions

For markets like:
- "Will MrBeast's next video get 100M views?"
- "Most viewed YouTube video this month?"
- "Will X creator hit Y subscribers by date?"

Uses YouTube Data API v3 to fetch:
- Channel statistics (subscribers, view counts)
- Recent video performance (views, likes, growth rate)
- Upload frequency patterns
- Historical averages
"""

import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from config.settings import Settings


@dataclass
class VideoStats:
    """Stats for a single YouTube video."""
    video_id: str
    title: str
    channel: str
    published_at: str
    views: int
    likes: int
    comments: int
    age_hours: float

    @property
    def views_per_hour(self) -> float:
        return self.views / max(self.age_hours, 1)

    @property
    def projected_views_30d(self) -> int:
        """Project views to 30 days using decay model."""
        if self.age_hours < 1:
            return self.views
        # YouTube view decay: ~60% of lifetime views in first 48 hours
        # Use power law: views(t) = a * t^b (b ≈ 0.3 for viral, 0.15 for normal)
        if self.views_per_hour > 500000:  # viral
            b = 0.3
        elif self.views_per_hour > 100000:  # strong
            b = 0.25
        else:
            b = 0.15
        ratio = (720 / max(self.age_hours, 1)) ** b  # 720 = 30 days in hours
        return int(self.views * ratio)


@dataclass
class ChannelStats:
    """Stats for a YouTube channel."""
    channel_id: str
    name: str
    subscribers: int
    total_views: int
    video_count: int
    avg_views_per_video: int
    recent_videos: List[VideoStats]

    @property
    def avg_recent_views(self) -> int:
        if not self.recent_videos:
            return self.avg_views_per_video
        return sum(v.views for v in self.recent_videos) // max(len(self.recent_videos), 1)


@dataclass
class YouTubeAnalysis:
    """Analysis result for a YouTube-related market."""
    market_question: str
    probability_estimate: float
    confidence: float
    reasoning: str
    channel_stats: Optional[ChannelStats]
    recent_videos: List[VideoStats]
    key_metric: str  # what metric matters (views, subs, etc.)
    threshold: Optional[int] = None  # target number from market question


# Channel IDs for known creators
KNOWN_CHANNELS = {
    "mrbeast": "UCX6OQ3DkcsbYNE6H8uQQuVA",
    "pewdiepie": "UC-lHJZR3Gqxm24_Vd_AJ5Yw",
    "tseries": "UCq-Fj5jknLsUf-MWSy4_brA",
    "markrober": "UCY1kMZp36IQSyNx_9h4mpCg",
    "mkbhd": "UCBcRF18a7Qf58cCRy5xuWwQ",
    "loganpaul": "UCG8rbF3g2AMX70yOd8vqIZg",
    "ksi": "UCGmnsW623G1r-Chmo5RB4Yw",
    "dream": "UCTkXRDQl0luXxVQrRQvWS6w",
}


class YouTubeAnalyzer:
    """Fetches YouTube data and analyzes video/creator markets."""

    def __init__(self):
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = 300  # 5 min

    async def analyze_market(self, market_question: str,
                              context: str = "") -> Optional[YouTubeAnalysis]:
        """Analyze a YouTube-related market question."""
        if not Settings.has_youtube():
            return self._estimate_without_api(market_question)

        question_lower = market_question.lower()

        # Detect which channel
        channel_id = None
        channel_name = ""
        for name, cid in KNOWN_CHANNELS.items():
            if name in question_lower:
                channel_id = cid
                channel_name = name
                break

        if not channel_id:
            return None

        # Fetch channel data
        channel_stats = await self._get_channel_stats(channel_id, channel_name)
        if not channel_stats:
            return None

        # Detect market type
        threshold = self._extract_number(market_question)

        if "views" in question_lower or "view" in question_lower:
            return self._analyze_views(market_question, channel_stats, threshold)
        elif "subscriber" in question_lower:
            return self._analyze_subscribers(market_question, channel_stats, threshold)
        else:
            return self._analyze_views(market_question, channel_stats, threshold)

    async def _get_channel_stats(self, channel_id: str,
                                  name: str) -> Optional[ChannelStats]:
        """Fetch channel + recent video statistics."""
        cache_key = f"channel:{channel_id}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[1] < self._cache_ttl:
            return cached[0]

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                # Get channel stats
                resp = await client.get(
                    "https://www.googleapis.com/youtube/v3/channels",
                    params={
                        "part": "statistics,snippet",
                        "id": channel_id,
                        "key": Settings.YOUTUBE_API_KEY,
                    }
                )
                if resp.status_code != 200:
                    return None

                ch_data = resp.json().get("items", [{}])[0]
                stats = ch_data.get("statistics", {})
                subscribers = int(stats.get("subscriberCount", 0))
                total_views = int(stats.get("viewCount", 0))
                video_count = int(stats.get("videoCount", 0))

                # Get recent videos
                search_resp = await client.get(
                    "https://www.googleapis.com/youtube/v3/search",
                    params={
                        "part": "snippet",
                        "channelId": channel_id,
                        "order": "date",
                        "maxResults": 10,
                        "type": "video",
                        "key": Settings.YOUTUBE_API_KEY,
                    }
                )
                recent_videos = []
                if search_resp.status_code == 200:
                    items = search_resp.json().get("items", [])
                    video_ids = [i["id"]["videoId"] for i in items if "videoId" in i.get("id", {})]

                    if video_ids:
                        vid_resp = await client.get(
                            "https://www.googleapis.com/youtube/v3/videos",
                            params={
                                "part": "statistics,snippet",
                                "id": ",".join(video_ids),
                                "key": Settings.YOUTUBE_API_KEY,
                            }
                        )
                        if vid_resp.status_code == 200:
                            for v in vid_resp.json().get("items", []):
                                v_stats = v.get("statistics", {})
                                snippet = v.get("snippet", {})
                                pub = snippet.get("publishedAt", "")
                                age_h = 999.0
                                try:
                                    dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                                    age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                                except (ValueError, TypeError):
                                    pass

                                recent_videos.append(VideoStats(
                                    video_id=v["id"],
                                    title=snippet.get("title", ""),
                                    channel=name,
                                    published_at=pub,
                                    views=int(v_stats.get("viewCount", 0)),
                                    likes=int(v_stats.get("likeCount", 0)),
                                    comments=int(v_stats.get("commentCount", 0)),
                                    age_hours=age_h,
                                ))

                result = ChannelStats(
                    channel_id=channel_id,
                    name=name,
                    subscribers=subscribers,
                    total_views=total_views,
                    video_count=video_count,
                    avg_views_per_video=total_views // max(video_count, 1),
                    recent_videos=sorted(recent_videos, key=lambda v: v.views, reverse=True),
                )
                self._cache[cache_key] = (result, time.time())
                return result

        except Exception as e:
            print(f"⚠️ YouTube API error: {e}")
            return None

    def _analyze_views(self, question: str, channel: ChannelStats,
                       threshold: Optional[int]) -> YouTubeAnalysis:
        """Analyze a views-based market."""
        if not threshold:
            threshold = 100_000_000  # default 100M

        # Use recent video performance to estimate
        if channel.recent_videos:
            latest = channel.recent_videos[0]
            projected = latest.projected_views_30d
            avg_recent = channel.avg_recent_views

            # Probability = how often recent videos exceed threshold
            exceed_count = sum(1 for v in channel.recent_videos if v.views >= threshold)
            base_prob = exceed_count / max(len(channel.recent_videos), 1)

            # Adjust based on latest video trajectory
            if latest.age_hours < 72:  # recent video, can project
                trajectory_prob = min(1.0, projected / threshold) if threshold > 0 else 0.5
                probability = (base_prob * 0.4 + trajectory_prob * 0.6)
            else:
                probability = base_prob

            confidence = 0.6 if len(channel.recent_videos) >= 5 else 0.4
            reasoning = (
                f"Based on {len(channel.recent_videos)} recent videos. "
                f"Average: {avg_recent:,} views. "
                f"Latest ({latest.title[:30]}): {latest.views:,} views in {latest.age_hours:.0f}h, "
                f"projected to {projected:,}. "
                f"{exceed_count}/{len(channel.recent_videos)} recent videos exceeded {threshold:,}."
            )
        else:
            probability = 0.5
            confidence = 0.2
            reasoning = "No recent video data available"

        return YouTubeAnalysis(
            market_question=question,
            probability_estimate=max(0.01, min(0.99, probability)),
            confidence=confidence,
            reasoning=reasoning,
            channel_stats=channel,
            recent_videos=channel.recent_videos[:5],
            key_metric="views",
            threshold=threshold,
        )

    def _analyze_subscribers(self, question: str, channel: ChannelStats,
                             threshold: Optional[int]) -> YouTubeAnalysis:
        """Analyze a subscriber-based market."""
        if not threshold:
            threshold = channel.subscribers + 10_000_000

        remaining = threshold - channel.subscribers
        if remaining <= 0:
            return YouTubeAnalysis(
                market_question=question,
                probability_estimate=0.95,
                confidence=0.9,
                reasoning=f"Already at {channel.subscribers:,} subs, above {threshold:,} target",
                channel_stats=channel,
                recent_videos=channel.recent_videos[:3],
                key_metric="subscribers",
                threshold=threshold,
            )

        # Estimate growth rate from channel data
        # Rough: assume ~0.5% monthly growth for large channels
        monthly_growth_rate = 0.005
        months_needed = remaining / (channel.subscribers * monthly_growth_rate)
        probability = max(0.05, min(0.95, 1.0 - (months_needed / 24)))

        return YouTubeAnalysis(
            market_question=question,
            probability_estimate=probability,
            confidence=0.3,
            reasoning=(
                f"Current: {channel.subscribers:,}. "
                f"Target: {threshold:,}. "
                f"Need {remaining:,} more. "
                f"Est. {months_needed:.0f} months at avg growth."
            ),
            channel_stats=channel,
            recent_videos=channel.recent_videos[:3],
            key_metric="subscribers",
            threshold=threshold,
        )

    def _estimate_without_api(self, question: str) -> Optional[YouTubeAnalysis]:
        """Rough estimate without YouTube API access."""
        question_lower = question.lower()
        if "mrbeast" not in question_lower:
            return None

        threshold = self._extract_number(question) or 100_000_000

        # MrBeast averages ~150-200M per video, 90%+ videos exceed 100M
        if threshold <= 100_000_000:
            prob = 0.85
        elif threshold <= 200_000_000:
            prob = 0.55
        elif threshold <= 300_000_000:
            prob = 0.25
        else:
            prob = 0.10

        return YouTubeAnalysis(
            market_question=question,
            probability_estimate=prob,
            confidence=0.4,
            reasoning=f"Estimated without API. MrBeast avg ~150-200M views. Threshold: {threshold:,}",
            channel_stats=None,
            recent_videos=[],
            key_metric="views",
            threshold=threshold,
        )

    @staticmethod
    def _extract_number(text: str) -> Optional[int]:
        """Extract a large number from text (e.g., '100M views', '50 million')."""
        import re

        # Try "100M", "100 million", etc.
        m = re.search(r'(\d+(?:\.\d+)?)\s*[Mm](?:illion)?', text)
        if m:
            return int(float(m.group(1)) * 1_000_000)

        m = re.search(r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?', text)
        if m:
            return int(float(m.group(1)) * 1_000_000_000)

        m = re.search(r'(\d+(?:\.\d+)?)\s*[Kk]', text)
        if m:
            return int(float(m.group(1)) * 1_000)

        # Plain number (>1000 to avoid matching dates)
        m = re.search(r'(\d{4,})', text.replace(",", ""))
        if m:
            return int(m.group(1))

        return None
