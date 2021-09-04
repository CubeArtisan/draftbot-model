#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <concurrentqueue.h>
#include <blockingconcurrentqueue.h>
#include <pcg_random.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

constexpr std::size_t MAX_IN_PACK = 24;
constexpr std::size_t MAX_SEEN = 400;
constexpr std::size_t MAX_PICKED = 48;
constexpr std::size_t NUM_LAND_COMBS = 8;

struct PyPick {
    // We manipulate it so the first card is always the one chosen to simplify the model's loss calculation.
    static constexpr std::int32_t chosen_card = 0;

    std::array<std::int32_t, MAX_IN_PACK> in_pack{0};
    std::array<std::int32_t, MAX_SEEN> seen{0};
    float num_seen{0.f};
    std::array<std::int32_t, MAX_PICKED> picked{0};
    float num_picked{0.f};
    std::array<std::array<std::int32_t, 2>, 4> coords{{{0, 0}}};
    std::array<float, 4> coord_weights{0.f};
    std::array<std::array<std::uint8_t, MAX_SEEN>, NUM_LAND_COMBS> seen_probs{{{0}}};
    std::array<std::array<std::uint8_t, MAX_PICKED>, NUM_LAND_COMBS> picked_probs{{{0}}};
    std::array<std::array<std::uint8_t, MAX_IN_PACK>, NUM_LAND_COMBS> in_pack_probs{{{0}}};

    bool load_from(std::ifstream& current_file) {
        std::array<char, (MAX_IN_PACK + MAX_SEEN + MAX_PICKED) * (NUM_LAND_COMBS + 2)> read_buffer;
        current_file.read(read_buffer.data(), 4 * 2 * sizeof(std::uint8_t) + 4 * sizeof(float) + 3 * sizeof(std::uint16_t));
        if (!current_file.good()) return false;
        const char* current_pos = read_buffer.data();
        *this = {};
        for (std::size_t i = 0; i < 4; i++) {
            for (std::size_t j = 0; j < 2; j++) {
                coords[i][j] = *reinterpret_cast<const std::uint8_t*>(current_pos);
                current_pos += sizeof(std::uint8_t);
            }
        }
        for (std::size_t i = 0; i < 4; i++) {
            coord_weights[i] = *reinterpret_cast<const float*>(current_pos);
            current_pos += sizeof(float);
        }
        std::uint16_t num_in_pack = *reinterpret_cast<const std::uint16_t*>(current_pos);
        current_pos += sizeof(std::uint16_t);
        num_picked = *reinterpret_cast<const std::uint16_t*>(current_pos);
        current_pos += sizeof(std::uint16_t);
        num_seen = *reinterpret_cast<const std::uint16_t*>(current_pos);
        current_file.read(read_buffer.data(), (num_in_pack + num_picked + num_seen) * (NUM_LAND_COMBS * sizeof(std::uint8_t) + sizeof(std::uint16_t)));
        if (!current_file.good()) return false;
        current_pos = read_buffer.data();
        for (std::int32_t i = 0; i < num_in_pack; i++) {
            in_pack[i] = *reinterpret_cast<const std::uint16_t*>(current_pos);
            current_pos += sizeof(std::uint16_t);
        }
        for (std::size_t i = 0; i < num_in_pack; i++) {
            for (std::size_t j = 0; j < NUM_LAND_COMBS; j++) {
                in_pack_probs[j][i] = *reinterpret_cast<const std::uint8_t*>(current_pos);
                current_pos += sizeof(std::uint8_t);
            }
        }
        for (std::int32_t i = 0; i < num_picked; i++) {
            picked[i] = *reinterpret_cast<const std::uint16_t*>(current_pos);
            current_pos += sizeof(std::uint16_t);
        }
        for (std::size_t i = 0; i < num_picked; i++) {
            for (std::size_t j = 0; j < NUM_LAND_COMBS; j++) {
                picked_probs[j][i] = *reinterpret_cast<const std::uint8_t*>(current_pos);
                current_pos += sizeof(std::uint8_t);
            }
        }
        for (std::int32_t i = 0; i < num_seen; i++) {
            seen[i] = *reinterpret_cast<const std::uint16_t*>(current_pos);
            current_pos += sizeof(std::uint16_t);
        }
        for (std::size_t i = 0; i < num_seen; i++) {
            for (std::size_t j = 0; j < NUM_LAND_COMBS; j++) {
                seen_probs[j][i] = *reinterpret_cast<const std::uint8_t*>(current_pos);
                current_pos += sizeof(std::uint8_t);
            }
        }
        return true;
    }
};

struct PyPickBatch {
    using python_type = std::tuple<py::array_t<std::int32_t>, py::array_t<std::int32_t>,
        py::array_t<float>, py::array_t<std::int32_t>, py::array_t<float>,
        py::array_t<std::int32_t>, py::array_t<float>, py::array_t<std::uint8_t>,
        py::array_t<std::uint8_t>, py::array_t<std::uint8_t>>;
    // using python_type = std::tuple<py::array_t<std::int32_t>, py::array_t<std::int32_t>,
    //     py::array_t<float>, py::array_t<std::int32_t>, py::array_t<float>,
    //     py::array_t<std::int32_t>, py::array_t<float>, py::array_t<float>,
    //     py::array_t<float>, py::array_t<float>>;

    constexpr std::array<std::size_t, 2> in_pack_shape() const noexcept { return { batch_size, MAX_IN_PACK }; }
    constexpr std::array<std::size_t, 2> seen_shape() const noexcept { return { batch_size, MAX_SEEN }; }
    constexpr std::array<std::size_t, 1> num_seen_shape() const noexcept { return { batch_size }; }
    constexpr std::array<std::size_t, 2> picked_shape() const noexcept { return { batch_size, MAX_PICKED }; }
    constexpr std::array<std::size_t, 1> num_picked_shape() const noexcept { return { batch_size }; }
    constexpr std::array<std::size_t, 3> coords_shape() const noexcept { return { batch_size, 4, 2 }; }
    constexpr std::array<std::size_t, 2> coord_weights_shape() const noexcept { return { batch_size, 4 }; }
    constexpr std::array<std::size_t, 3> seen_probs_shape() const noexcept { return { batch_size, NUM_LAND_COMBS, MAX_SEEN }; }
    constexpr std::array<std::size_t, 3> picked_probs_shape() const noexcept { return { batch_size, NUM_LAND_COMBS, MAX_PICKED }; }
    constexpr std::array<std::size_t, 3> in_pack_probs_shape() const noexcept { return { batch_size, NUM_LAND_COMBS, MAX_IN_PACK }; }

    static constexpr std::array<std::size_t, 2> in_pack_strides{ sizeof(PyPick), sizeof(std::int32_t) };
    static constexpr std::array<std::size_t, 2> seen_strides{ sizeof(PyPick), sizeof(std::int32_t) };
    static constexpr std::array<std::size_t, 1> num_seen_strides{ sizeof(PyPick) };
    static constexpr std::array<std::size_t, 2> picked_strides{ sizeof(PyPick), sizeof(std::int32_t) };
    static constexpr std::array<std::size_t, 1> num_picked_strides{ sizeof(PyPick) };
    static constexpr std::array<std::size_t, 3> coords_strides{ sizeof(PyPick), sizeof(std::array<std::int32_t, 2>), sizeof(std::int32_t) };
    static constexpr std::array<std::size_t, 2> coord_weights_strides{ sizeof(PyPick), sizeof(float) };
    // static constexpr std::array<std::size_t, 3> seen_probs_strides{ sizeof(PyPick), sizeof(std::array<float, MAX_SEEN>), sizeof(float) };
    // static constexpr std::array<std::size_t, 3> picked_probs_strides{ sizeof(PyPick), sizeof(std::array<float, MAX_PICKED>), sizeof(float) };
    // static constexpr std::array<std::size_t, 3> in_pack_probs_strides{ sizeof(PyPick), sizeof(std::array<float, MAX_IN_PACK>), sizeof(float) };
    static constexpr std::array<std::size_t, 3> seen_probs_strides{ sizeof(PyPick), sizeof(std::array<std::uint8_t, MAX_SEEN>), sizeof(std::uint8_t) };
    static constexpr std::array<std::size_t, 3> picked_probs_strides{ sizeof(PyPick), sizeof(std::array<std::uint8_t, MAX_PICKED>), sizeof(std::uint8_t) };
    static constexpr std::array<std::size_t, 3> in_pack_probs_strides{ sizeof(PyPick), sizeof(std::array<std::uint8_t, MAX_IN_PACK>), sizeof(std::uint8_t) };

    constexpr std::size_t size() const noexcept { return batch_size; }

    auto begin() noexcept { return storage.begin(); }
    auto begin() const noexcept { return storage.begin(); }
    auto end() noexcept { return storage.end(); }
    auto end() const noexcept { return storage.end(); }

    PyPickBatch(std::size_t picks_per_batch)
            : batch_size(picks_per_batch), storage(batch_size)
    { }

    PyPickBatch(PyPickBatch&& other)
            : batch_size(other.batch_size), storage(std::move(other.storage))
    { }

    PyPickBatch& operator=(PyPickBatch&& other) {
        batch_size = other.batch_size;
        storage = std::move(other.storage);
        return *this;
    }

    PyPick& operator[](std::size_t index) {
        return storage[index];
    }

    python_type to_numpy(py::capsule& capsule) noexcept {
        PyPick& pick = storage.front();
        return {
            py::array_t<std::int32_t>(in_pack_shape(), in_pack_strides, reinterpret_cast<std::int32_t*>(pick.in_pack.data()), capsule),
            py::array_t<std::int32_t>(seen_shape(), seen_strides, reinterpret_cast<std::int32_t*>(pick.seen.data()), capsule),
            py::array_t<float>(num_seen_shape(), num_seen_strides, reinterpret_cast<float*>(&pick.num_seen), capsule),
            py::array_t<std::int32_t>(picked_shape(), picked_strides, reinterpret_cast<std::int32_t*>(pick.picked.data()), capsule),
            py::array_t<float>(num_picked_shape(), num_picked_strides, reinterpret_cast<float*>(&pick.num_picked), capsule),
            py::array_t<std::int32_t>(coords_shape(), coords_strides, reinterpret_cast<std::int32_t*>(pick.coords.data()), capsule),
            py::array_t<float>(coord_weights_shape(), coord_weights_strides, reinterpret_cast<float*>(pick.coord_weights.data()), capsule),
            py::array_t<std::uint8_t>(seen_probs_shape(), seen_probs_strides, reinterpret_cast<std::uint8_t*>(pick.seen_probs.data()), capsule),
            py::array_t<std::uint8_t>(picked_probs_shape(), picked_probs_strides, reinterpret_cast<std::uint8_t*>(pick.picked_probs.data()), capsule),
            py::array_t<std::uint8_t>(in_pack_probs_shape(), in_pack_probs_strides, reinterpret_cast<std::uint8_t*>(pick.in_pack_probs.data()), capsule)
        };
    }

    python_type to_numpy(py::capsule& capsule) const noexcept {
        const PyPick& pick = storage.front();
        return {
            py::array_t<std::int32_t>(in_pack_shape(), in_pack_strides, reinterpret_cast<const std::int32_t*>(pick.in_pack.data()), capsule),
            py::array_t<std::int32_t>(seen_shape(), seen_strides, reinterpret_cast<const std::int32_t*>(pick.seen.data()), capsule),
            py::array_t<float>(num_seen_shape(), num_seen_strides, reinterpret_cast<const float*>(&pick.num_seen), capsule),
            py::array_t<std::int32_t>(picked_shape(), picked_strides, reinterpret_cast<const std::int32_t*>(pick.picked.data()), capsule),
            py::array_t<float>(num_picked_shape(), num_picked_strides, reinterpret_cast<const float*>(&pick.num_picked), capsule),
            py::array_t<std::int32_t>(coords_shape(), coords_strides, reinterpret_cast<const std::int32_t*>(pick.coords.data()), capsule),
            py::array_t<float>(coord_weights_shape(), coord_weights_strides, reinterpret_cast<const float*>(pick.coord_weights.data()), capsule),
            py::array_t<std::uint8_t>(seen_probs_shape(), seen_probs_strides, reinterpret_cast<const std::uint8_t*>(pick.seen_probs.data()), capsule),
            py::array_t<std::uint8_t>(picked_probs_shape(), picked_probs_strides, reinterpret_cast<const std::uint8_t*>(pick.picked_probs.data()), capsule),
            py::array_t<std::uint8_t>(in_pack_probs_shape(), in_pack_probs_strides, reinterpret_cast<const std::uint8_t*>(pick.in_pack_probs.data()), capsule)
        };
    }

private:
    std::size_t batch_size;
    std::vector<PyPick> storage;
};

struct DraftPickGenerator {
    using result_type = typename PyPickBatch::python_type;
    static constexpr std::size_t read_buffer_count = (1ull << 16) / sizeof(PyPick); // 64 KB
    static constexpr std::size_t buffered_pick_count = (1ull << 32) / sizeof(PyPick); // 4 GB
    static constexpr std::size_t shuffle_buffer_count = (1ull << 28) / sizeof(PyPick); // 256 MB

    DraftPickGenerator(std::size_t picks_per_batch, std::size_t num_workers,
                       std::size_t seed, const std::string& folder_path)
            : batch_size(picks_per_batch), num_threads{num_workers},
              initial_seed{seed}, length{0}, loaded_batches{buffered_pick_count / picks_per_batch},
              files_to_read_producer{files_to_read}, loaded_batches_consumer{loaded_batches},
              main_rng{initial_seed, num_workers} {
        py::gil_scoped_release release;
        std::cout << "\tbuffered_pick_count: " << buffered_pick_count << " in batches: " << buffered_pick_count / batch_size
                  << "\n\tshuffle_buffer_count: " << shuffle_buffer_count
                  << "\n\tread_buffer_count: " << read_buffer_count << std::endl;
        std::vector<char> loaded_file_buffer;
        for (const auto& path_data : std::filesystem::directory_iterator(folder_path)) {
            draft_filenames.push_back(path_data.path().string());
            std::ifstream picks_file(path_data.path(), std::ios::binary | std::ios::ate);
            auto file_size = picks_file.tellg();
            loaded_file_buffer.clear();
            loaded_file_buffer.resize(file_size);
            picks_file.seekg(0);
            picks_file.read(loaded_file_buffer.data(), file_size);
            const char* current_pos = loaded_file_buffer.data();
            const char* const end_pos = loaded_file_buffer.data() + loaded_file_buffer.size();
            const std::size_t prev_length = length;
            while (current_pos < end_pos) {
                length++;
                current_pos = skip_record(current_pos, end_pos);
            }
        }
    }

    DraftPickGenerator& enter() {
        py::gil_scoped_release release;
        if (exit_threads) {
            exit_threads = false;
            queue_new_epoch();
            delayed_loads = buffered_pick_count + num_threads * shuffle_buffer_count;
            delayed_batches = buffered_pick_count / batch_size;
            request_needed_work();
            std::size_t thread_number = 0;
            for (std::size_t i=0; i < num_threads; i++) {
                worker_threads.emplace_back([this, j=thread_number++](){ this->worker(pcg32(this->initial_seed, j)); });
            }
        }
        return *this;
    }

    bool exit(py::object, py::object, py::object) {
        exit_threads = true;
        for (auto& worker : worker_threads) worker.join();
        return false;
    }

    std::size_t size() const noexcept { return (length + batch_size - 1) / batch_size; }

    DraftPickGenerator& queue_new_epoch() {
        std::shuffle(std::begin(draft_filenames), std::end(draft_filenames), main_rng);
        files_to_read.enqueue_bulk(files_to_read_producer, std::begin(draft_filenames), draft_filenames.size());
        return *this;
    }

    result_type next() {
        std::unique_ptr<PyPickBatch> batched;
        if (!loaded_batches.try_dequeue(loaded_batches_consumer, batched)) {
            py::gil_scoped_release gil_release;
            do {
                request_needed_work();
                std::cout << "\nloaded_batches: " << loaded_batches.size_approx()
                          << ", loaded_picks: " << loaded_picks.size_approx()
                          << ", requested_tasks: " << requested_tasks.size_approx()
                          << ", files_to_read: " << files_to_read.size_approx()
                          << std::endl;
            } while (!loaded_batches.wait_dequeue_timed(loaded_batches_consumer, batched, 100'000));
        }
        delayed_loads += batch_size;
        delayed_batches += 1;
        request_needed_work();
        PyPickBatch* batched_ptr = batched.release();
        py::capsule free_when_done(batched_ptr, [](void* ptr) { delete reinterpret_cast<PyPickBatch*>(ptr); });
        return batched_ptr->to_numpy(free_when_done);
    }

    result_type getitem(std::size_t) { return next(); }

private:
    void request_needed_work() {
        std::size_t required_read_tasks = delayed_loads / read_buffer_count;
        delayed_loads -= required_read_tasks * read_buffer_count;
        std::size_t required_batch_tasks = delayed_batches;
        delayed_batches -= required_batch_tasks;
        std::vector<int> requests(required_batch_tasks + required_read_tasks, 0);
        std::uniform_int_distribution<std::size_t> index_selector(0, requests.size() - 1);
        for (std::size_t i=0; i < required_batch_tasks; i++) {
            std::size_t index = index_selector(main_rng);
            if (requests[index] == 0) {
                requests[index] = 1; 
            } else {
                required_batch_tasks += 1;
            }
        }
        requested_tasks.enqueue_bulk(requests.begin(), requests.size());
		if (files_to_read.size_approx() <= num_threads) queue_new_epoch();
    }

    static const char* skip_record(const char* current_pos, const char* end_pos) noexcept {
        current_pos += sizeof(std::array<std::array<std::uint8_t, 2>, 4>) + sizeof(std::array<float, 4>);
        if (current_pos + 3 * sizeof(std::uint16_t) > end_pos) return end_pos + 1;
        std::uint16_t num_in_pack = *reinterpret_cast<const std::uint16_t*>(current_pos);
        current_pos += sizeof(std::uint16_t);
        std::uint16_t num_picked = *reinterpret_cast<const std::uint16_t*>(current_pos);
        current_pos += sizeof(std::uint16_t);
        std::uint16_t num_seen = *reinterpret_cast<const std::uint16_t*>(current_pos);
        current_pos += sizeof(std::uint16_t);
        return current_pos + (sizeof(std::uint16_t) + sizeof(std::array<std::uint8_t, NUM_LAND_COMBS>))
                              * (num_in_pack + num_picked + num_seen);
    }

    void worker(pcg32 rng) {
        std::vector<PyPick> picks_buffer(read_buffer_count);
        std::uniform_int_distribution<std::size_t> sleep_for(1'000, 39'000);
        moodycamel::ConsumerToken files_to_read_consumer(files_to_read);
        moodycamel::ProducerToken loaded_picks_producer(loaded_picks);
        moodycamel::ConsumerToken requested_tasks_consumer(requested_tasks);
        moodycamel::ConsumerToken loaded_picks_consumer(loaded_picks);
        moodycamel::ProducerToken loaded_batches_producer(loaded_batches);
        moodycamel::ProducerToken requested_tasks_producer(requested_tasks);
        std::vector<PyPick> shuffle_buffer;
        shuffle_buffer.reserve(shuffle_buffer_count);
        std::optional<std::ifstream> current_file;
        std::size_t current_index = 0;
        std::unique_ptr<PyPickBatch> batch = std::make_unique<PyPickBatch>(batch_size);
        auto iter = batch->begin();
        auto end_iter = batch->end();
        std::uniform_int_distribution<std::size_t> index_selector(0, shuffle_buffer_count - 1);
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_for(rng)));
        while (!exit_threads) {
            const char* current_pos;
            const char* end_pos;
            int task = 0;
            while (!exit_threads && !requested_tasks.try_dequeue(requested_tasks_consumer, task)) {
                std::this_thread::sleep_for(std::chrono::microseconds(sleep_for(rng)));
            }
            if (exit_threads) break;
            if (task == 0) {
                while (current_index < picks_buffer.size() && !exit_threads) {
                    if (!current_file || !current_file->good() || current_file->eof()) {                        
                        std::string cur_filename;
                        if (files_to_read.try_dequeue_from_producer(files_to_read_producer, cur_filename)) {
                            current_file = std::ifstream(cur_filename, std::ios::binary);
                        } else {
                            current_file = std::nullopt;
                            queue_new_epoch();
                        }
                        delayed_loads += read_buffer_count;
                        break;
                    }
                    if (!picks_buffer[current_index++].load_from(*current_file)) {
                        current_file = std::nullopt;
                        current_index -= 1;
                        delayed_loads += read_buffer_count;
                        break;
                    }
                }
                if (current_index >= picks_buffer.size()) {
                    loaded_picks.enqueue_bulk(loaded_picks_producer, std::begin(picks_buffer), picks_buffer.size());
                    current_index = 0;
                }
            } else if (task == 1) {
                if (shuffle_buffer.capacity() > shuffle_buffer.size()) {
                    loaded_picks.try_dequeue_bulk(loaded_picks_consumer, std::back_inserter(shuffle_buffer),
                                                  shuffle_buffer.capacity() - shuffle_buffer.size());
                    delayed_batches++;
                    continue;
                }
                iter += loaded_picks.try_dequeue_bulk(loaded_picks_consumer, iter,
                                                      std::distance(iter, end_iter));
                if (iter == end_iter) {
                    for (std::size_t i = 0; i < batch->size(); i++) {
                        std::size_t swap_index = index_selector(rng);
                        if (swap_index >= shuffle_buffer.size()) std::cout << swap_index << std::endl;
                        std::swap(batch->operator[](i), shuffle_buffer[swap_index]);
                    }
                    loaded_batches.enqueue(loaded_batches_producer, std::move(batch));
                    batch = std::make_unique<PyPickBatch>(batch_size);
                    iter = batch->begin();
                    end_iter = batch->end();
                } else {
                    delayed_batches++;
                }
            }
        }
    }

    std::size_t batch_size;
    std::size_t num_threads;
    std::size_t initial_seed;

    std::vector<std::string> draft_filenames;
    std::size_t length;

    moodycamel::ConcurrentQueue<std::string> files_to_read;
    moodycamel::ConcurrentQueue<PyPick> loaded_picks;
    moodycamel::ConcurrentQueue<int> requested_tasks;
    moodycamel::BlockingConcurrentQueue<std::unique_ptr<PyPickBatch>> loaded_batches;
    moodycamel::ProducerToken files_to_read_producer;
    moodycamel::ConsumerToken loaded_batches_consumer;

    std::atomic<bool> exit_threads{true};
    std::atomic<std::size_t> delayed_loads{ 0 };
    std::atomic<std::size_t> delayed_batches{ 0 };
    std::vector<std::thread> worker_threads;

    pcg32 main_rng;
};

PYBIND11_MODULE(draftbot_generator, m) {
    using namespace pybind11::literals;
    py::class_<DraftPickGenerator>(m, "DraftPickGenerator")
        .def(py::init<std::size_t, std::size_t, std::size_t, const std::string&>())
        .def("__enter__", &DraftPickGenerator::enter)
        .def("__exit__", &DraftPickGenerator::exit)
        .def("__len__", &DraftPickGenerator::size)
        .def("__getitem__", &DraftPickGenerator::getitem)
        .def("__next__", &DraftPickGenerator::next)
        .def("__iter__", &DraftPickGenerator::queue_new_epoch)
        .def("on_epoch_end", &DraftPickGenerator::queue_new_epoch)
        .def("queue_new_epoch", &DraftPickGenerator::queue_new_epoch);
}
