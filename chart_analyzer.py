import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')


class MLChartPatternDetector:
    """
    Enhanced chart pattern detector using machine learning algorithms
    Focuses on overall stock patterns rather than small segments
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.pattern_templates = self._create_pattern_templates()

    def _create_pattern_templates(self):
        """Create normalized templates for overall chart patterns"""
        templates = {}

        # Long-term trend patterns (more points for overall analysis)

        # Bullish Trend - steady upward movement
        templates['bullish_trend'] = np.array([0.1, 0.15, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        # Bearish Trend - steady downward movement
        templates['bearish_trend'] = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.15, 0.1])

        # Sideways/Consolidation - horizontal movement
        templates['sideways_consolidation'] = np.array([0.5, 0.52, 0.48, 0.51, 0.49, 0.5, 0.48, 0.52, 0.49, 0.51])

        # V-shaped Recovery - sharp decline followed by sharp recovery
        templates['v_recovery'] = np.array([0.8, 0.6, 0.4, 0.2, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9])

        # Inverted V - sharp rise followed by sharp decline
        templates['inverted_v'] = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1])

        # Cup pattern - decline, bottom, gradual recovery
        templates['cup_pattern'] = np.array([0.8, 0.7, 0.5, 0.3, 0.2, 0.2, 0.3, 0.5, 0.7, 0.8])

        # Rounding Top - gradual rise then gradual decline
        templates['rounding_top'] = np.array([0.2, 0.3, 0.5, 0.7, 0.8, 0.8, 0.7, 0.5, 0.3, 0.2])

        # Rounding Bottom - gradual decline then gradual rise
        templates['rounding_bottom'] = np.array([0.8, 0.7, 0.5, 0.3, 0.2, 0.2, 0.3, 0.5, 0.7, 0.8])

        # Breakout pattern - consolidation followed by strong move up
        templates['bullish_breakout'] = np.array([0.5, 0.51, 0.49, 0.5, 0.48, 0.52, 0.6, 0.7, 0.8, 0.9])

        # Breakdown pattern - consolidation followed by strong move down
        templates['bearish_breakdown'] = np.array([0.5, 0.49, 0.51, 0.5, 0.52, 0.48, 0.4, 0.3, 0.2, 0.1])

        # Channel Up - consistent higher highs and higher lows
        templates['ascending_channel'] = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.4, 0.35, 0.5, 0.45, 0.6])

        # Channel Down - consistent lower highs and lower lows
        templates['descending_channel'] = np.array([0.9, 0.8, 0.85, 0.7, 0.75, 0.6, 0.65, 0.5, 0.55, 0.4])

        # Volatile sideways - high volatility with no clear direction
        templates['volatile_sideways'] = np.array([0.5, 0.7, 0.3, 0.6, 0.4, 0.8, 0.2, 0.7, 0.3, 0.5])

        # Normalize all templates to 0-1 range
        for name, template in templates.items():
            templates[name] = self._normalize_segment(template)
        return templates

    def _normalize_segment(self, segment):
        """Normalize a price segment to a 0-1 range."""
        if len(segment) == 0:
            return np.array([])
        min_val = np.min(segment)
        max_val = np.max(segment)
        if max_val == min_val:
            return np.zeros_like(segment)
        return (segment - min_val) / (max_val - min_val)

    def _prepare_overall_pattern(self, prices, num_points=10):
        """
        Prepare the overall price pattern by sampling key points across the entire dataset
        """
        if len(prices) < num_points:
            return prices

        # Create evenly spaced indices across the entire price series
        indices = np.linspace(0, len(prices) - 1, num_points, dtype=int)
        sampled_prices = prices[indices]

        return sampled_prices

    def _calculate_trend_strength(self, prices):
        """Calculate the strength and direction of the overall trend"""
        if len(prices) < 2:
            return 0, "neutral"

        # Use linear regression to determine trend
        x = np.arange(len(prices)).reshape(-1, 1)
        y = prices.reshape(-1, 1)

        reg = LinearRegression().fit(x, y)
        slope = reg.coef_[0][0]
        r_squared = reg.score(x, y)

        # Normalize slope by price range to get relative trend strength
        price_range = np.max(prices) - np.min(prices)
        if price_range > 0:
            trend_strength = abs(slope) / price_range * len(prices)
        else:
            trend_strength = 0

        # Determine trend direction
        if slope > 0 and r_squared > 0.3:
            direction = "bullish"
        elif slope < 0 and r_squared > 0.3:
            direction = "bearish"
        else:
            direction = "sideways"

        return trend_strength * r_squared, direction

    def _calculate_volatility_metrics(self, prices):
        """Calculate volatility and stability metrics"""
        if len(prices) < 2:
            return 0, 0

        # Calculate returns
        returns = np.diff(prices) / prices[:-1]

        # Volatility (standard deviation of returns)
        volatility = np.std(returns)

        # Stability (inverse of coefficient of variation)
        mean_price = np.mean(prices)
        if mean_price > 0:
            stability = 1 / (np.std(prices) / mean_price)
        else:
            stability = 0

        return volatility, stability

    def _detect_overall_pattern_match(self, prices):
        """
        Match the overall price pattern against templates
        """
        # Prepare the overall pattern (sample key points)
        overall_pattern = self._prepare_overall_pattern(prices, num_points=10)
        normalized_overall = self._normalize_segment(overall_pattern)

        best_matches = {}

        for pattern_name, template in self.pattern_templates.items():
            if len(normalized_overall) != len(template):
                continue

            try:
                # Calculate cosine similarity for overall pattern
                score = cosine_similarity(
                    normalized_overall.reshape(1, -1),
                    template.reshape(1, -1)
                )[0][0]

                if score > 0.7:  # Threshold for overall pattern matching
                    best_matches[pattern_name] = {
                        'score': score,
                        'position': 0,  # Overall pattern starts from beginning
                        'template_len': len(prices),  # Covers entire dataset
                        'normalized_template': template,
                        'original_segment_prices': overall_pattern,
                        'confidence': score  # Add confidence directly
                    }
            except ValueError:
                continue

        return best_matches

    def _identify_key_levels(self, prices):
        """
        Identify key support and resistance levels using multiple methods
        """
        if len(prices) < 20:
            return np.nan, np.nan

        try:
            current_price = float(prices[-1])

            # Method 1: Find significant highs and lows using peak detection
            min_distance = max(10, len(prices) // 20)  # Minimum distance between peaks

            # Find peaks (potential resistance)
            peaks, peak_properties = find_peaks(prices, distance=min_distance, prominence=np.std(prices) * 0.5)

            # Find troughs (potential support)
            troughs, trough_properties = find_peaks(-prices, distance=min_distance, prominence=np.std(prices) * 0.5)

            # Method 2: Calculate pivot points from recent highs and lows
            recent_period = min(50, len(prices) // 2)
            recent_prices = prices[-recent_period:]
            recent_high = np.max(recent_prices)
            recent_low = np.min(recent_prices)

            # Method 3: Calculate support and resistance levels
            support_candidates = []
            resistance_candidates = []

            # From peaks and troughs
            if len(troughs) > 0:
                trough_prices = prices[troughs]
                # Only consider troughs below current price for support
                valid_troughs = trough_prices[trough_prices <= current_price]
                if len(valid_troughs) > 0:
                    support_candidates.extend(valid_troughs)

            if len(peaks) > 0:
                peak_prices = prices[peaks]
                # Only consider peaks above current price for resistance
                valid_peaks = peak_prices[peak_prices >= current_price]
                if len(valid_peaks) > 0:
                    resistance_candidates.extend(valid_peaks)

            # Add recent highs and lows as candidates
            if recent_low <= current_price:
                support_candidates.append(recent_low)
            if recent_high >= current_price:
                resistance_candidates.append(recent_high)

            # Add historical significant levels
            # Look for price levels that were tested multiple times
            price_levels = np.round(prices, 2)  # Round to nearest cent
            unique_levels, counts = np.unique(price_levels, return_counts=True)

            # Find levels that were tested at least 3 times
            significant_levels = unique_levels[counts >= 3]

            for level in significant_levels:
                if level <= current_price:
                    support_candidates.append(level)
                elif level >= current_price:
                    resistance_candidates.append(level)

            # Calculate final support and resistance
            if support_candidates:
                # Take the highest support level (closest to current price)
                support_candidates = [s for s in support_candidates if s <= current_price]
                if support_candidates:
                    support = max(support_candidates)
                else:
                    support = recent_low * 0.95  # Fallback
            else:
                # Fallback: use recent low with some buffer
                support = recent_low * 0.95

            if resistance_candidates:
                # Take the lowest resistance level (closest to current price)
                resistance_candidates = [r for r in resistance_candidates if r >= current_price]
                if resistance_candidates:
                    resistance = min(resistance_candidates)
                else:
                    resistance = recent_high * 1.05  # Fallback
            else:
                # Fallback: use recent high with some buffer
                resistance = recent_high * 1.05

            # Ensure support is below current price and resistance is above
            if support >= current_price:
                support = current_price * 0.95
            if resistance <= current_price:
                resistance = current_price * 1.05

            # Additional validation: ensure reasonable spread
            price_range = np.max(prices) - np.min(prices)
            min_spread = price_range * 0.02  # Minimum 2% of total range

            if (current_price - support) < min_spread:
                support = current_price - min_spread
            if (resistance - current_price) < min_spread:
                resistance = current_price + min_spread

            return float(support), float(resistance)

        except Exception as e:
            print(f"Error calculating support/resistance: {e}")
            # Fallback calculation
            try:
                current_price = float(prices[-1])
                price_std = np.std(prices)
                support = current_price - (price_std * 1.5)
                resistance = current_price + (price_std * 1.5)
                return float(support), float(resistance)
            except:
                return np.nan, np.nan

    def _generate_recommendation(self, pattern_name, trend_strength, trend_direction, volatility, current_vs_support,
                                 current_vs_resistance):
        """Generate trading recommendation based on overall analysis"""

        # Base recommendation from pattern
        pattern_recommendations = {
            'bullish_trend': 'buy',
            'bearish_trend': 'sell',
            'sideways_consolidation': 'hold',
            'v_recovery': 'buy',
            'inverted_v': 'sell',
            'cup_pattern': 'buy',
            'rounding_top': 'sell',
            'rounding_bottom': 'buy',
            'bullish_breakout': 'strong buy',
            'bearish_breakdown': 'strong sell',
            'ascending_channel': 'buy',
            'descending_channel': 'sell',
            'volatile_sideways': 'hold'
        }

        base_recommendation = pattern_recommendations.get(pattern_name, 'hold')

        # Adjust based on trend strength and other factors
        if trend_strength > 0.5:
            if trend_direction == 'bullish' and base_recommendation in ['buy', 'hold']:
                return 'strong buy'
            elif trend_direction == 'bearish' and base_recommendation in ['sell', 'hold']:
                return 'strong sell'

        # Consider position relative to support/resistance
        if current_vs_support < 0.05 and base_recommendation in ['buy', 'hold']:
            return 'buy'  # Near support
        elif current_vs_resistance < 0.05 and base_recommendation in ['sell', 'hold']:
            return 'sell'  # Near resistance

        return base_recommendation

    def detect_patterns(self, prices, volumes=None):
        """
        Main method to detect overall chart patterns and provide recommendations.
        Returns:
            tuple: (pattern_name, recommendation, support, resistance, pattern_info)
        """
        prices_array = np.array(prices).flatten().astype(float)
        if len(prices_array) == 0:
            return "No Data", "hold", np.nan, np.nan, {}

        # Calculate support and resistance
        support, resistance = self._identify_key_levels(prices_array)

        # Calculate trend metrics
        trend_strength, trend_direction = self._calculate_trend_strength(prices_array)
        volatility, stability = self._calculate_volatility_metrics(prices_array)

        # Detect overall pattern matches
        pattern_matches = self._detect_overall_pattern_match(prices_array)

        best_pattern_name = "No clear pattern"
        best_recommendation = "hold"
        best_pattern_info = {}

        if pattern_matches:
            # Get the best matching pattern
            best_match = max(pattern_matches.items(), key=lambda x: x[1]['score'])
            best_pattern_name = best_match[0]
            best_pattern_info = best_match[1]

            # Calculate position relative to support/resistance
            current_price = prices_array[-1]
            current_vs_support = abs(current_price - support) / current_price if not np.isnan(support) else 1
            current_vs_resistance = abs(current_price - resistance) / current_price if not np.isnan(resistance) else 1

            # Generate recommendation
            best_recommendation = self._generate_recommendation(
                best_pattern_name, trend_strength, trend_direction,
                volatility, current_vs_support, current_vs_resistance
            )
        else:
            # Fallback to trend-based recommendation if no pattern matches
            if trend_strength > 0.3:
                if trend_direction == 'bullish':
                    best_recommendation = 'buy'
                    best_pattern_name = 'Bullish Trend'
                elif trend_direction == 'bearish':
                    best_recommendation = 'sell'
                    best_pattern_name = 'Bearish Trend'
                else:
                    best_recommendation = 'hold'
                    best_pattern_name = 'Sideways Movement'

            # Create pattern info for trend-based patterns
            if best_pattern_name != "No clear pattern":
                best_pattern_info = {
                    'score': trend_strength,
                    'position': 0,
                    'template_len': len(prices_array),
                    'normalized_template': self._normalize_segment(prices_array),
                    'original_segment_prices': prices_array,
                    'confidence': min(trend_strength, 0.9)
                }

        # Volume confirmation (if available)
        if volumes is not None and len(volumes) > 0 and best_pattern_name != "No clear pattern":
            volumes_array = np.array(volumes).flatten()
            if len(volumes_array) > 10:
                # Check if recent volume supports the pattern
                recent_volume_avg = np.mean(volumes_array[-10:])
                historical_volume_avg = np.mean(volumes_array[:-10]) if len(volumes_array) > 10 else recent_volume_avg

                volume_ratio = recent_volume_avg / historical_volume_avg if historical_volume_avg > 0 else 1

                # Strengthen recommendation if volume confirms
                if volume_ratio > 1.2:  # 20% higher volume
                    if best_recommendation == 'buy':
                        best_recommendation = 'strong buy'
                    elif best_recommendation == 'sell':
                        best_recommendation = 'strong sell'

        # Format pattern name for display
        display_pattern_name = best_pattern_name.replace('_',
                                                         ' ').title() if best_pattern_name != "No clear pattern" else "No clear pattern"

        return display_pattern_name, best_recommendation, support, resistance, best_pattern_info


# Global instance
ml_detector = MLChartPatternDetector()


def detect_chart_pattern(prices, volumes=None):
    """
    Enhanced chart pattern detection focusing on overall stock patterns
    """
    try:
        return ml_detector.detect_patterns(prices, volumes)
    except Exception as e:
        print(f"Error in ML pattern detection: {e}")
        # Calculate basic support/resistance even when pattern detection fails
        try:
            prices_array = np.array(prices, dtype=np.float64).flatten()
            if len(prices_array) > 0:
                current_price = float(prices_array[-1])
                price_std = np.std(prices_array)
                support = current_price - (price_std * 1.5)
                resistance = current_price + (price_std * 1.5)
                return "Error in detection", "hold", support, resistance, {}
            else:
                return "Error in detection", "hold", 0.0, 0.0, {}
        except:
            return "Error in detection", "hold", 0.0, 0.0, {}