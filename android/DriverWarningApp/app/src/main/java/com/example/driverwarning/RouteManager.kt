package com.example.driverwarning

import android.util.Log
import com.google.android.gms.maps.model.LatLng
import kotlinx.coroutines.*
import org.json.JSONObject
import java.net.URL

/**
 * Fetches a planned route from the Google Directions API, decodes the
 * overview polyline, and resamples it to ~10 m point spacing for
 * downstream curve analysis.
 *
 * Usage:
 *   val rm = RouteManager()
 *   rm.fetchRoute(origin, destination, apiKey) { success ->
 *       if (success) {
 *           val polyline = rm.getPolyline()
 *       }
 *   }
 */
class RouteManager {

    companion object {
        private const val TAG = "RouteManager"
        private const val DIRECTIONS_BASE =
            "https://maps.googleapis.com/maps/api/directions/json"
        private const val RESAMPLE_SPACING_M = 10.0   // metres between resampled points
    }

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    @Volatile private var routePolyline: List<LatLng> = emptyList()
    @Volatile private var isFetching = false

    fun getPolyline(): List<LatLng> = routePolyline

    fun hasRoute(): Boolean = routePolyline.size >= 3

    fun clearRoute() {
        routePolyline = emptyList()
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Fetch a driving route asynchronously.
     *
     * @param origin      "lat,lon" string or address.
     * @param destination "lat,lon" string or address.
     * @param apiKey      Google Maps / Directions API key.
     * @param scope       CoroutineScope to launch the network request in.
     * @param onResult    Called on success (true) or failure (false) on the
     *                    main thread.
     */
    fun fetchRoute(
        origin: String,
        destination: String,
        apiKey: String,
        scope: CoroutineScope,
        onResult: (success: Boolean) -> Unit,
    ) {
        if (isFetching) {
            Log.w(TAG, "Route fetch already in progress; skipping.")
            return
        }
        isFetching = true
        Log.d(TAG, "Fetching route: $origin → $destination")

        scope.launch(Dispatchers.IO) {
            val result = runCatching {
                val url = buildUrl(origin, destination, apiKey)
                val json = URL(url).readText()
                val obj = JSONObject(json)
                val status = obj.getString("status")
                if (status != "OK") {
                    Log.e(TAG, "Directions API status: $status")
                    return@runCatching emptyList()
                }
                val encoded = obj
                    .getJSONArray("routes")
                    .getJSONObject(0)
                    .getJSONObject("overview_polyline")
                    .getString("points")

                val raw = decodePolyline(encoded)
                resamplePolyline(raw, RESAMPLE_SPACING_M)
            }

            val polyline = result.getOrElse {
                Log.e(TAG, "Route fetch exception", it)
                emptyList()
            }

            routePolyline = polyline
            isFetching = false

            val success = polyline.size >= 3
            Log.d(TAG, "Route loaded: ${polyline.size} points (success=$success)")

            withContext(Dispatchers.Main) {
                onResult(success)
            }
        }
    }

    // -------------------------------------------------------------------------
    // Google encoded polyline decoder
    // -------------------------------------------------------------------------

    /**
     * Decode a Google encoded polyline string into a list of [LatLng].
     *
     * Reference:
     * https://developers.google.com/maps/documentation/utilities/polylinealgorithm
     */
    fun decodePolyline(encoded: String): List<LatLng> {
        val result = mutableListOf<LatLng>()
        var index = 0
        var lat = 0
        var lng = 0

        while (index < encoded.length) {
            // Decode latitude
            var b: Int
            var shift = 0
            var value = 0
            do {
                b = encoded[index++].code - 63
                value = value or ((b and 0x1F) shl shift)
                shift += 5
            } while (b >= 0x20)
            val dLat = if ((value and 1) != 0) -(value shr 1) else (value shr 1)
            lat += dLat

            // Decode longitude
            shift = 0
            value = 0
            do {
                b = encoded[index++].code - 63
                value = value or ((b and 0x1F) shl shift)
                shift += 5
            } while (b >= 0x20)
            val dLng = if ((value and 1) != 0) -(value shr 1) else (value shr 1)
            lng += dLng

            result.add(LatLng(lat / 1e5, lng / 1e5))
        }
        return result
    }

    // -------------------------------------------------------------------------
    // Polyline resampling
    // -------------------------------------------------------------------------

    /**
     * Resample a polyline so consecutive points are ≈ [spacingM] metres apart.
     * Uniform density ensures the curve scanner gets consistent curvature
     * estimates regardless of Directions API point density.
     */
    fun resamplePolyline(
        points: List<LatLng>,
        spacingM: Double = RESAMPLE_SPACING_M,
    ): List<LatLng> {
        if (points.size < 2) return points

        val resampled = mutableListOf(points.first())
        var accumulated = 0.0

        for (i in 1 until points.size) {
            val p1 = points[i - 1]
            val p2 = points[i]
            val segLen = haversineM(p1.latitude, p1.longitude, p2.latitude, p2.longitude)
            if (segLen < 1e-6) continue

            var remaining = segLen
            var pos = p1

            while (accumulated + remaining >= spacingM) {
                val frac = (spacingM - accumulated) / remaining
                val newPt = LatLng(
                    pos.latitude  + frac * (p2.latitude  - pos.latitude),
                    pos.longitude + frac * (p2.longitude - pos.longitude),
                )
                resampled.add(newPt)
                remaining -= (spacingM - accumulated)
                accumulated = 0.0
                pos = newPt
            }
            accumulated += remaining
        }

        if (resampled.last() != points.last()) resampled.add(points.last())
        return resampled
    }

    // -------------------------------------------------------------------------
    // Geometry helpers
    // -------------------------------------------------------------------------

    fun haversineM(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val R = 6_371_000.0
        val phi1 = Math.toRadians(lat1)
        val phi2 = Math.toRadians(lat2)
        val dPhi = Math.toRadians(lat2 - lat1)
        val dLam = Math.toRadians(lon2 - lon1)
        val a = Math.sin(dPhi / 2).pow2() +
                Math.cos(phi1) * Math.cos(phi2) * Math.sin(dLam / 2).pow2()
        return R * 2 * Math.asin(Math.sqrt(a))
    }

    private fun Double.pow2() = this * this

    // -------------------------------------------------------------------------
    // URL builder
    // -------------------------------------------------------------------------

    private fun buildUrl(origin: String, destination: String, apiKey: String): String {
        val enc = { s: String -> java.net.URLEncoder.encode(s, "UTF-8") }
        return "$DIRECTIONS_BASE?origin=${enc(origin)}&destination=${enc(destination)}" +
               "&mode=driving&key=${enc(apiKey)}"
    }
}
