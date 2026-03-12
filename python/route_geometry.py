"""
route_geometry.py

Utilities for fetching a planned route from the Google Directions API,
decoding the encoded polyline, and resampling it to a uniform point spacing
for downstream curve analysis.

Usage:
    from python.route_geometry import fetch_route_polyline, resample_polyline
"""

import math
import json
from typing import List, Tuple, Optional
try:
    import urllib.request as urlrequest
except ImportError:
    urlrequest = None


# ---------------------------------------------------------------------------
# Polyline encoding / decoding
# ---------------------------------------------------------------------------

def decode_polyline(encoded: str) -> List[Tuple[float, float]]:
    """
    Decode a Google encoded polyline string into a list of (lat, lon) tuples.

    Algorithm reference:
      https://developers.google.com/maps/documentation/utilities/polylinealgorithm

    Args:
        encoded: Google encoded polyline string.

    Returns:
        List of (latitude, longitude) float tuples.
    """
    coords: List[Tuple[float, float]] = []
    index = 0
    lat = 0
    lon = 0

    while index < len(encoded):
        # Decode latitude
        result = 0
        shift = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        # Decode longitude
        result = 0
        shift = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlon = ~(result >> 1) if (result & 1) else (result >> 1)
        lon += dlon

        coords.append((lat / 1e5, lon / 1e5))

    return coords


def encode_polyline(coords: List[Tuple[float, float]]) -> str:
    """
    Encode a list of (lat, lon) pairs into a Google encoded polyline string.
    Useful for round-trip testing.
    """
    def _encode_value(v: int) -> str:
        # Standard Google algorithm:
        # 1. Left-shift by 1; if negative, invert all bits
        e = (~(v << 1)) if v < 0 else (v << 1)
        chunks = []
        while e >= 0x20:
            chunks.append(chr((0x20 | (e & 0x1F)) + 63))
            e >>= 5
        chunks.append(chr(e + 63))
        return "".join(chunks)

    output = []
    prev_lat = 0
    prev_lon = 0
    for lat, lon in coords:
        lat_e5 = round(lat * 1e5)
        lon_e5 = round(lon * 1e5)
        output.append(_encode_value(lat_e5 - prev_lat))
        output.append(_encode_value(lon_e5 - prev_lon))
        prev_lat = lat_e5
        prev_lon = lon_e5
    return "".join(output)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in metres between two GPS points."""
    R = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _interpolate(p1: Tuple[float, float], p2: Tuple[float, float], frac: float) -> Tuple[float, float]:
    """Linear interpolation between two lat/lon points."""
    return (
        p1[0] + frac * (p2[0] - p1[0]),
        p1[1] + frac * (p2[1] - p1[1]),
    )


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_polyline(
    coords: List[Tuple[float, float]],
    spacing_m: float = 10.0,
) -> List[Tuple[float, float]]:
    """
    Resample a polyline so that consecutive points are approximately
    ``spacing_m`` metres apart. This ensures uniform density for curvature
    scanning regardless of how the Directions API returned the points.

    Args:
        coords:    List of (lat, lon) tuples from the raw route.
        spacing_m: Target spacing in metres (default 10 m).

    Returns:
        Resampled list of (lat, lon) tuples.
    """
    if len(coords) < 2:
        return list(coords)

    resampled: List[Tuple[float, float]] = [coords[0]]
    accumulated = 0.0

    for i in range(1, len(coords)):
        p1 = coords[i - 1]
        p2 = coords[i]
        seg_len = _haversine_m(*p1, *p2)

        if seg_len < 1e-6:
            continue

        remaining = seg_len
        pos = p1

        while accumulated + remaining >= spacing_m:
            frac = (spacing_m - accumulated) / remaining
            new_point = _interpolate(pos, p2, frac)
            resampled.append(new_point)

            # Advance along the segment
            traveled = _haversine_m(*pos, *new_point)
            remaining -= (spacing_m - accumulated)
            accumulated = 0.0
            pos = new_point

        accumulated += remaining

    # Always include the final point
    if resampled[-1] != coords[-1]:
        resampled.append(coords[-1])

    return resampled


# ---------------------------------------------------------------------------
# Google Directions API fetching
# ---------------------------------------------------------------------------

def fetch_route_polyline(
    origin: str,
    destination: str,
    api_key: str,
    mode: str = "driving",
    spacing_m: float = 10.0,
) -> Optional[List[Tuple[float, float]]]:
    """
    Fetch a route from the Google Directions API and return a resampled
    list of (lat, lon) waypoints.

    Args:
        origin:      Origin address or "lat,lon" string.
        destination: Destination address or "lat,lon" string.
        api_key:     Google Maps / Directions API key.
        mode:        Travel mode (default "driving").
        spacing_m:   Resampling spacing in metres.

    Returns:
        Resampled list of (lat, lon) tuples, or None on failure.
    """
    import urllib.parse
    import urllib.request

    base = "https://maps.googleapis.com/maps/api/directions/json"
    params = urllib.parse.urlencode({
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "key": api_key,
    })
    url = f"{base}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[route_geometry] API request failed: {exc}")
        return None

    if data.get("status") != "OK":
        print(f"[route_geometry] Directions API status: {data.get('status')}")
        return None

    try:
        encoded = data["routes"][0]["overview_polyline"]["points"]
    except (KeyError, IndexError) as exc:
        print(f"[route_geometry] Could not extract polyline: {exc}")
        return None

    raw = decode_polyline(encoded)
    return resample_polyline(raw, spacing_m=spacing_m)


# ---------------------------------------------------------------------------
# CLI utility for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test route geometry utilities")
    parser.add_argument("--origin", type=str, help="Origin (address or lat,lon)")
    parser.add_argument("--destination", type=str, help="Destination")
    parser.add_argument("--api-key", type=str, default="", help="Google API key")
    parser.add_argument("--spacing", type=float, default=10.0, help="Resample spacing (m)")
    args = parser.parse_args()

    if args.origin and args.destination and args.api_key:
        pts = fetch_route_polyline(args.origin, args.destination, args.api_key, spacing_m=args.spacing)
        if pts:
            print(f"Route fetched: {len(pts)} points @ {args.spacing}m spacing")
            print(f"  First: {pts[0]}")
            print(f"  Last:  {pts[-1]}")
        else:
            print("Route fetch failed.")
    else:
        # Simple decode/encode round-trip test
        sample = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
        decoded = decode_polyline(sample)
        reenc = encode_polyline(decoded)
        print(f"Decoded {len(decoded)} points from sample polyline")
        print(f"  First: {decoded[0]}")
        print(f"Re-encoded matches: {reenc == sample}")
        resampled = resample_polyline(decoded, spacing_m=50)
        print(f"Resampled at 50m: {len(resampled)} points")
