import unittest
from unittest.mock import Mock, patch

from amap_client import amap_transit_route, find_balanced_center
from app_v2 import _validate_participants, app, calculate_routes


class MultiParticipantTest(unittest.TestCase):
    def setUp(self):
        self.participants = [
            {"id": "p1", "name": "甲", "location": {"lng": 116.0, "lat": 39.0, "name": "A"}, "preference": "transit"},
            {"id": "p2", "name": "乙", "location": {"lng": 117.0, "lat": 40.0, "name": "B"}, "preference": "driving"},
            {"id": "p3", "name": "丙", "location": {"lng": 118.0, "lat": 41.0, "name": "C"}, "preference": "walking"},
        ]

    def test_validates_two_to_eight_participants(self):
        self.assertEqual(_validate_participants(self.participants), self.participants)
        with self.assertRaisesRegex(ValueError, "2-8"):
            _validate_participants(self.participants[:1])
        with self.assertRaisesRegex(ValueError, "2-8"):
            _validate_participants(self.participants * 3)

    def test_balanced_center_uses_all_participants(self):
        result = find_balanced_center(self.participants)
        self.assertEqual(result["midpoint"], {"lng": 117.0, "lat": 40.0})
        self.assertGreaterEqual(result["suggested_search_radius_m"], 500)

    @patch("amap_client._amap_get")
    def test_transit_v5_forwards_strategy_and_parses_cost_duration(self, amap_get):
        response = Mock()
        response.json.return_value = {
            "status": "1",
            "route": {
                "transits": [{
                    "cost": {"duration": "900", "transit_fee": "3.0"},
                    "distance": "8000",
                    "walking_distance": "500",
                    "segments": [{
                        "walking": {"distance": "500", "cost": {"duration": "300"}},
                        "bus": {"buslines": [{"name": "地铁6号线", "distance": "7500"}]},
                    }],
                }],
            },
        }
        amap_get.return_value = response

        result = amap_transit_route(
            116.4, 39.9, 116.5, 39.95,
            city="北京", strategy=7,
        )

        url = amap_get.call_args.args[0]
        params = amap_get.call_args.kwargs["params"]
        self.assertIn("/v5/direction/transit/integrated", url)
        self.assertEqual(params["strategy"], 7)
        self.assertEqual(params["city1"], "110000")
        self.assertEqual(result["duration_minutes"], 15)
        self.assertIn("地铁6号线", result["line_summary"])

    def test_uses_current_deepseek_model(self):
        import app_v2
        self.assertEqual(app_v2.DEEPSEEK_MODEL, "deepseek-v4-flash")

    def test_search_api_rejects_legacy_two_location_payload(self):
        response = app.test_client().post("/api/v2/search", json={
            "location_a": self.participants[0]["location"],
            "location_b": self.participants[1]["location"],
            "query": "餐厅",
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("2-8", response.get_json()["error"])

    @patch("app_v2.amap_get_best_route")
    def test_calculate_routes_aggregates_every_participant(self, get_route):
        get_route.side_effect = [
            {"success": True, "mode": "transit", "duration_minutes": 10, "duration_text": "10分钟"},
            {"success": True, "mode": "driving", "duration_minutes": 20, "duration_text": "20分钟"},
            {"success": True, "mode": "walking", "duration_minutes": 35, "duration_text": "35分钟"},
        ]
        pois = [{"name": "会合点", "lng": 116.5, "lat": 39.5, "rating": 4.5}]

        result = calculate_routes(pois, self.participants, city="北京")

        self.assertEqual(len(result[0]["routes"]), 3)
        self.assertEqual(result[0]["total_time_minutes"], 65)
        self.assertEqual(result[0]["time_range_minutes"], 25)


    def test_non_transit_mode_ignores_transit_strategy_requirements(self):
        participants = [
            {**self.participants[0], "preference": "driving", "transit_strategy": 6},
            self.participants[1],
        ]

        self.assertEqual(_validate_participants(participants), participants)

    @patch("app_v2.amap_get_best_route")
    def test_route_retry_preserves_transit_strategy_and_poi_ids(self, get_route):
        participant = {
            "id": "p1",
            "name": "甲",
            "location": {
                "lng": 116.0,
                "lat": 39.0,
                "name": "A",
                "poi_id": "origin-poi",
            },
            "preference": "transit",
            "transit_strategy": 7,
        }
        poi = {"id": "destination-poi", "name": "会合点", "lng": 116.5, "lat": 39.5, "rating": 4.5}
        get_route.side_effect = [
            {"success": False, "mode": "transit"},
            {"success": True, "mode": "transit", "strategy": 7, "duration_minutes": 10, "duration_text": "10分钟"},
        ]

        result = calculate_routes([poi], [participant], city="北京")

        retry_args = get_route.call_args_list[1].args
        self.assertEqual(retry_args[7:], (7, "origin-poi", "destination-poi"))
        self.assertEqual(result[0]["routes"][0]["strategy"], 7)


if __name__ == "__main__":
    unittest.main()
