void print_exchange_resplit_timing(const std::string& tag,
                                      const double old_ref_seconds,
                                      const double rs_ref_seconds,
                                      const double old_mp_seconds,
                                      const double rs_mp_seconds)
{
    std::stringstream ss;
    ss << "=== " << tag << " exchange resplit timing ===\n"
       << "  old original = " << old_ref_seconds * 1.0e3 << " ms\n"
       << "  RS original  = " << rs_ref_seconds * 1.0e3 << " ms\n"
       << "  old MP       = " << old_mp_seconds * 1.0e3 << " ms\n"
       << "  RS MP        = " << rs_mp_seconds * 1.0e3 << " ms\n"
       << "  original RS speedup = " << old_ref_seconds / rs_ref_seconds << "\n"
       << "  MP RS speedup       = " << old_mp_seconds / rs_mp_seconds << "\n"
       << "===============================\n";
    write_to_ablation_file(ss.str());
}
