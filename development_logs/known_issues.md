# Known Issues

1. **Volume of Vision Assets:**
   - The Vision dataset contains over 23,600 JPG images. The scanner streams and verifies headers correctly, but initial bulk indexing takes 1-2 minutes on standard HDD/SSD storage.
2. **Path Normalization:**
   - All relative paths are stored in POSIX format (`/`) to ensure deterministic JSON schemas across Windows and Linux environments.
