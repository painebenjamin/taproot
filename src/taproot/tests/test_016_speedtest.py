from typing import Dict, List

from taproot import Server
from taproot.util import (
    blue,
    cyan,
    execute_echo_test,
    generate_temp_key_and_cert,
    get_test_server_addresses,
    green,
    human_duration,
    human_size,
    plot_echo_test_results,
    random_ascii_string,
    save_test_image,
    AsyncRunner,
    ServerSubprocessRunner,
)

def test_echo_speed() -> None:
    async def run_test() -> None:
        key, cert = generate_temp_key_and_cert()
        encryption_key = random_ascii_string(32).encode("utf-8")

        results: Dict[str, Dict[int, List[float]]] = {}
        for address in get_test_server_addresses(no_memory=True):
            server = Server()
            server.address = address
            server.keyfile = key
            server.certfile = cert
            server.encryption_key = encryption_key
            client = server.get_client()
            async with ServerSubprocessRunner(server):
                results[server.scheme] = await execute_echo_test(client)
    
        for scheme, data in results.items():
            print(f"{cyan(scheme)}:")
            for size, times in data.items():
                min_time = min(times)
                max_time = max(times)
                median_time = sorted(times)[len(times) // 2]
                mean_time = sum(times) / len(times)
                transfer_rate = size * 2 / mean_time
                print(f"  {blue(human_size(size, precision=0))}")
                print(f"    min: {green(human_duration(min_time))}")
                print(f"    max: {green(human_duration(max_time))}")
                print(f"    median: {green(human_duration(median_time))}")
                print(f"    mean: {green(human_duration(mean_time))}")
                print(f"    transfer rate: {green(human_size(transfer_rate) + '/s')}")
        print(
            save_test_image(
                plot_echo_test_results(results),
                subject="echo_test"
            )
        )

    AsyncRunner(run_test).run()
